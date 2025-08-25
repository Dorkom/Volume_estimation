#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prueba SOLO la ALTURA por bloque usando:
  • Imagen normal
  • JSON de Roboflow con 'bloque'
  • height_calibrator.joblib (MiDaS features -> H_cm)

Flujo:
  1) Carga imagen y JSON
  2) (Importante) Redimensiona la imagen a (image.width, image.height) del JSON
  3) Ejecuta MiDaS sobre esa imagen
  4) Para cada bbox (izq→der), extrae features de profundidad y predice H_cm

Uso:
python probar_altura_midas.py ^
  --img  "C:\ruta\frame.jpg" ^
  --json "C:\ruta\resultado.json" ^
  --calib "C:\...\height_calibrator.joblib" ^
  --device cuda --draw out.png
"""

import argparse, json, sys, math
from pathlib import Path

import numpy as np
import cv2
import torch
import joblib

# ----------------- MiDaS -----------------

def load_midas(model_type="DPT_Hybrid", device="cuda"):
    dev = torch.device("cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu")
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(dev).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform if "DPT" in model_type else transforms.small_transform
    return midas, transform, dev

def run_midas_depth(midas, transform, device, img_rgb):
    inp = transform(img_rgb).to(device)
    with torch.no_grad():
        pred = midas(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    return pred.detach().cpu().numpy().astype(np.float32)

# --------- Features de profundidad por bbox ---------

def depth_feats_from_bbox(depth, bbox, img_h, img_w, band_frac=0.15):
    x, y, w, h = bbox
    x0 = max(0, int(math.floor(x))); y0 = max(0, int(math.floor(y)))
    x1 = min(img_w, int(math.ceil(x+w))); y1 = min(img_h, int(math.ceil(h+y)))
    if x1 <= x0 or y1 <= y0:
        return None
    roi = depth[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    h_roi = roi.shape[0]
    band = max(1, int(round(h_roi * band_frac)))
    top_band    = roi[:band, :]
    bottom_band = roi[-band:, :]

    top_med = float(np.median(top_band))
    bot_med = float(np.median(bottom_band))
    delta   = bot_med - top_med
    ratio   = bot_med / (top_med + 1e-6)
    iqr     = float(np.percentile(roi, 75) - np.percentile(roi, 25))

    h_px = float(h)
    cx = x + 0.5*w
    cy = y + 0.5*h
    y_base = y + h

    feats = {
        "h_px": h_px,
        "top_med": top_med,
        "bot_med": bot_med,
        "delta": delta,
        "ratio": ratio,
        "iqr": iqr,
        "cy_n": cy/(img_h+1e-6),
        "ybase_n": y_base/(img_h+1e-6)
    }
    return feats

# ---------------- Utils JSON ----------------

def order_left_to_right(preds):
    for p in preds:
        p["_x_center"] = float(p["x"])
    preds.sort(key=lambda d: d["_x_center"])
    return preds

def bbox_from_pred(p):
    # Roboflow: x,y = centro; width,height = tamaño
    cx, cy = float(p["x"]), float(p["y"])
    w, h   = float(p["width"]), float(p["height"])
    x = cx - w/2
    y = cy - h/2
    return (x, y, w, h)

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img",   required=True, help="Imagen original (no calibrada)")
    ap.add_argument("--json",  required=True, help="JSON de Roboflow (predictions)")
    ap.add_argument("--calib", required=True, help="height_calibrator.joblib")
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--draw",  default=None, help="Ruta para guardar overlay con alturas (png/jpg)")
    args = ap.parse_args()

    # Cargar calibrador
    cal = joblib.load(args.calib)
    model = cal["model"]
    feat_names = cal["features"]
    band_frac  = float(cal.get("band_frac", 0.15))
    midas_type = cal.get("midas_type", "DPT_Hybrid")

    # Cargar JSON de Roboflow
    try:
        data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    except Exception as e:
        sys.exit(f"❌ JSON: {e}")
    preds = [p for p in data.get("predictions", []) if p.get("class")=="bloque"]
    if not preds:
        sys.exit("⚠️ No se encontraron 'bloque' en el JSON.")
    preds = order_left_to_right(preds)

    # Tamaño del JSON
    try:
        Wj, Hj = int(data["image"]["width"]), int(data["image"]["height"])
    except Exception:
        sys.exit("❌ El JSON no contiene image.width/height.")

    # Cargar imagen y AJUSTAR a (Wj, Hj)
    img_bgr = cv2.imread(args.img)
    if img_bgr is None:
        sys.exit(f"❌ No se pudo cargar la imagen: {args.img}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (Wj, Hj), interpolation=cv2.INTER_LINEAR)  # clave
    H_img, W_img = img_rgb.shape[:2]

    # Cargar MiDaS
    print(f"[INFO] MiDaS={midas_type}  device={args.device}")
    midas, midas_tf, device = load_midas(midas_type, args.device)
    depth = run_midas_depth(midas, midas_tf, device, img_rgb)

    # Visual opcional
    vis = img_rgb.copy() if args.draw else None

    print(f"\nBloques detectados: {len(preds)} (izq→der)\n")
    for i, p in enumerate(preds):
        bbox = bbox_from_pred(p)
        feats = depth_feats_from_bbox(depth, bbox, H_img, W_img, band_frac=band_frac)
        if feats is None:
            print(f"[Bloque {i}] ROI inválida para MiDaS → omitido.")
            continue

        X = np.array([[feats[n] for n in feat_names]], dtype=float)
        H_cm = float(model.predict(X)[0])

        print(f"[Bloque {i}]  Alto(MiDaS-cal) ≈ {H_cm:.1f} cm")

        # Dibujar si se pidió
        if vis is not None:
            x, y, w, h = bbox
            x0, y0 = int(round(x)), int(round(y))
            x1, y1 = int(round(x+w)), int(round(y+h))
            cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,0), 2)
            cv2.putText(vis, f"H~{H_cm:.1f}cm", (x0, max(0,y0-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)

    if vis is not None:
        out_path = Path(args.draw)
        out_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), out_bgr)
        print(f"\n🖼️ Overlay guardado en: {out_path}")

if __name__ == "__main__":
    main()
