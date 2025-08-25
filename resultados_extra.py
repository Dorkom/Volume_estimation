# resultados_extra.py / evaluar_modelos.py
import argparse, json, sys, math
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import joblib


# ============ Utils ============
def bbox_from_pred(p):
    cx, cy = float(p["x"]), float(p["y"])
    w, h   = float(p["width"]), float(p["height"])
    x = cx - w/2
    y = cy - h/2
    return x, y, w, h

def bbox_features_from_pred(p, img_W, img_H):
    x, y, w, h = bbox_from_pred(p)
    A = w * h
    aspect = w / h if h > 0 else 0.0
    cx = x + 0.5*w
    cy = y + 0.5*h
    y_base = y + h
    eps = 1e-6
    feats = [
        w, h, A, aspect,            # 4
        cx, cy, y_base,             # 3
        w/(img_W+eps), h/(img_H+eps),
        cx/(img_W+eps), cy/(img_H+eps),
        y_base/(img_H+eps),
        A/(img_W*img_H+eps)         # total 13
    ]
    return np.array([feats], dtype=float)

def order_left_to_right(preds):
    for p in preds:
        p["_x_center"] = float(p["x"])
    preds.sort(key=lambda d: d["_x_center"])
    return preds


# ============ ResNetRegressor (igual que en entrenamiento) ============
class ResNetRegressor(nn.Module):
    def __init__(self, backbone_name="resnet18", out_features=1):
        super().__init__()
        if backbone_name == "resnet18":
            self.backbone = torchvision.models.resnet18(weights=None)
            in_feats = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name == "resnet50":
            self.backbone = torchvision.models.resnet50(weights=None)
            in_feats = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("Backbone no soportado: 'resnet18'|'resnet50'")
        self.head = nn.Sequential(
            nn.Linear(in_feats, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, out_features),
        )
    def forward(self, x):
        z = self.backbone(x)
        y = self.head(z)            # predice V^(1/3)
        return y.squeeze(1)


# ============ Transform ============
def build_transform(img_size=224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img",  required=True, help="Ruta a imagen original")
    ap.add_argument("--json", required=True, help="JSON de Roboflow (predictions)")
    ap.add_argument("--cnn",  required=True, help="Checkpoint IA-CNN (best_resnet18_cuberoot.pth)")
    ap.add_argument("--ridge", required=True, help="Modelo Ridge (joblib) entrenado para V")
    ap.add_argument("--rf",    required=True, help="Modelo RandomForest (joblib) entrenado para V")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--pad", type=float, default=0.12, help="padding relativo del recorte")
    args = ap.parse_args()

    # Cargar JSON
    try:
        data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    except Exception as e:
        sys.exit(f"❌ JSON: {e}")

    preds = [p for p in data.get("predictions", []) if p.get("class")=="bloque"]
    if not preds:
        sys.exit("⚠️ No se encontraron 'bloque' en el JSON.")
    preds = order_left_to_right(preds)

    # Tamaño reportado por Roboflow
    try:
        Wj, Hj = int(data["image"]["width"]), int(data["image"]["height"])
    except Exception:
        sys.exit("❌ El JSON no contiene image.width/height.")

    # Cargar imagen y alinear al espacio del JSON
    img_cv = cv2.imread(args.img)
    if img_cv is None:
        sys.exit(f"❌ No se pudo cargar la imagen: {args.img}")
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (Wj, Hj))  # ¡clave para que coincidan!

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === IA-CNN: reconstruir exactamente la red de entrenamiento ===
    model = ResNetRegressor("resnet18").to(device)
    ckpt = torch.load(args.cnn, map_location=device)
    state = ckpt.get("model_state", ckpt)   # soporta ambos formatos
    model.load_state_dict(state, strict=True)
    model.eval()
    tfm = build_transform(args.img_size)

    # Modelos tabulares
    ridge = joblib.load(args.ridge)
    rf    = joblib.load(args.rf)

    print(f"Device: {device}")
    print(f"Bloques detectados: {len(preds)} (izq→der)\n")

    # Inferencia por bloque
    for i, p in enumerate(preds):
        cx, cy = float(p["x"]), float(p["y"])
        w, h   = float(p["width"]), float(p["height"])

        # recorte con padding relativo
        pad_w = w * args.pad
        pad_h = h * args.pad
        x0 = int(max(0, math.floor(cx - w/2 - pad_w)))
        y0 = int(max(0, math.floor(cy - h/2 - pad_h)))
        x1 = int(min(Wj, math.ceil(cx + w/2 + pad_w)))
        y1 = int(min(Hj, math.ceil(cy + h/2 + pad_h)))

        crop = img_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            print(f"[Bloque {i}] ⚠️ recorte vacío, omitido")
            continue

        # --- IA-CNN (V^(1/3) → V m³) ---
        x_t = tfm(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            y_cbr = model(x_t)                # V^(1/3)
            v_ai  = torch.clamp(y_cbr, min=0).pow(3).item()  # V (m³)
        print(f"[Bloque {i}]")
        print(f"  IA-CNN (ResNet18 → V^(1/3)):  V̂ = {v_ai:.3f} m³")

        # --- Tabular (13 features de bbox) ---
        X_vec = bbox_features_from_pred(p, Wj, Hj)   # shape (1,13)
        try:
            v_ridge = float(ridge.predict(X_vec)[0])
            print(f"  Ridge (tabular, V):           V̂ = {v_ridge:.3f} m³")
        except Exception as e:
            print(f"  Ridge: error al predecir → {e}")

        try:
            v_rf = float(rf.predict(X_vec)[0])
            print(f"  RandomForest (tabular, V):    V̂ = {v_rf:.3f} m³")
        except Exception as e:
            print(f"  RandomForest: error al predecir → {e}")

        print("")

if __name__ == "__main__":
    main()
