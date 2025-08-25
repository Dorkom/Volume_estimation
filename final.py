# calcular_dimensiones_v2.py
"""
Versión 2 – mayo 2025
 • 5 keypoints por bloque (new-point-0 … 4)
 • clase 'bloque'
 • largo, ancho y alto según pares 0-1, 1-2, 1-3
 • altura con factor de escala fijo
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient

# Ruta de la imagen original (ajusta según tu estructura)
IMAGE_PATH = "frame_000001.jpg"  # <-- REEMPLAZA ESTO

# ---------- parámetros globales ----------
DEFAULT_JSON  = "resultado.json"
DEFAULT_HOMOG = "homografia_tolva_inv.npy"

ESCALA_V_CM_PX = 125 / 33            # ← ajusta aquí si cambia la referencia

PAIR_LARGO = ("new-point-0", "new-point-1")   # m → cm
PAIR_ANCHO = ("new-point-1", "new-point-2")   # m → cm
PAIR_ALTO  = ("new-point-1", "new-point-3")   # px → cm
# -----------------------------------------

def transformar(px, py, Hinv):
    """píxel → mundo (m)"""
    p = np.array([[[px, py]]], dtype=np.float32)
    return cv2.perspectiveTransform(p, Hinv)[0, 0]   # (X, Y)

def kp_dict(pred):
    """dict etiqueta → (x, y)"""
    return {k["class"]: (k["x"], k["y"]) for k in pred["keypoints"]}

def run_roboflow_inference(image_path, model_id="etiquetado-bloques-keypoints/3", api_key="KOYZaBjvJzk174XAygWU"):
    """Run Roboflow inference and return prediction JSON."""
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )
    try:
        result = client.infer(image_path, model_id=model_id)
        with open("resultado.json", "w") as f:
            json.dump(result, f)
        print("✅ Resultado guardado en resultado.json")
        return result
    except Exception as e:
        sys.exit(f"❌ Error de inferencia: {e}")

def main():
    ap = argparse.ArgumentParser(description="Dimensiones de bloques (versión 2)")
    ap.add_argument("--homog", default=DEFAULT_HOMOG, help="homografia_tolva_inv.npy")
    args = ap.parse_args()

    # --- carga de archivos ---
    try:
        Hinv = np.load(args.homog)
    except Exception as e:
        sys.exit(f"❌ Homografía: {e}")

    print(f"🔍 Ejecutando inferencia en: {IMAGE_PATH}")
    data = run_roboflow_inference(IMAGE_PATH)

    preds = [p for p in data.get("predictions", []) if p.get("class") == "bloque"]
    if not preds:
        sys.exit("⚠️  No se encontraron objetos 'bloque'.")

    print("\nBloques detectados:")

    # --- Dibujar imagen con bounding boxes y keypoints ---
    try:
        image = cv2.imread(IMAGE_PATH)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar {IMAGE_PATH}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i, pred in enumerate(preds):
            x0 = int(pred["x"] - pred["width"]/2)
            y0 = int(pred["y"] - pred["height"]/2)
            x1 = int(pred["x"] + pred["width"]/2)
            y1 = int(pred["y"] + pred["height"]/2)
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(image, f"[{i}]", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            for kp in pred.get("keypoints", []):
                x, y = int(kp["x"]), int(kp["y"])
                label = kp["class"]
                cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(image, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        plt.figure(figsize=(12, 6))
        plt.imshow(image)
        plt.title("Bloques detectados con keypoints")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error al cargar o dibujar imagen: {e}")

    for i, p in enumerate(preds):
        print(f"  [{i}]  conf={p.get('confidence',0):.3f}  bbox={p['width']}×{p['height']}")

    sel = input("\n📝 Índices de bloques (ej. 0 1 2): ").strip()
    if not sel:
        sys.exit("Sin selección.")
    try:
        idxs = [int(s) for s in sel.replace(',', ' ').split()]
    except ValueError:
        sys.exit("Índices no válidos.")

    # --- cálculo ---
    necesarios = (*PAIR_LARGO, *PAIR_ANCHO, *PAIR_ALTO)
    for idx in idxs:
        if idx < 0 or idx >= len(preds):
            print(f"• {idx} fuera de rango – omitido.")
            continue

        kps = kp_dict(preds[idx])
        if not all(k in kps for k in necesarios):
            print(f"• Bloque {idx}: faltan keypoints – omitido.")
            continue

        pA0, pA1 = (transformar(*kps[k], Hinv) for k in PAIR_LARGO)
        pB0, pB1 = (transformar(*kps[k], Hinv) for k in PAIR_ANCHO)
        largo_cm = np.linalg.norm(pA0 - pA1) * 100
        ancho_cm = np.linalg.norm(pB0 - pB1) * 100

        y1, y3 = kps[PAIR_ALTO[0]][1], kps[PAIR_ALTO[1]][1]
        alto_cm = abs(y3 - y1) * ESCALA_V_CM_PX

        vol_cm3 = largo_cm * ancho_cm * alto_cm

        print(f"\n📦 Bloque {idx}")
        print(f"   Largo : {largo_cm:8.2f} cm")
        print(f"   Ancho : {ancho_cm:8.2f} cm")
        print(f"   Alto  : {alto_cm:8.2f} cm")
        print(f"   Volumen: {vol_cm3:,.0f} cm³")

if __name__ == "__main__":
    main()
