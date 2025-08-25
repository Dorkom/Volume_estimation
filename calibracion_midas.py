#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Entrena un CALIBRADOR de altura (MiDaS → H_real en cm) usando:
  • Excel con alturas reales (columna: 'ALTO PLANTA')
  • COCO _annotations.coco.json (para ubicar bboxes por imagen)
  • Imágenes normales (sin warp) en uno o más directorios raíz

Empareja por:
  - basename de la imagen (extraído de la columna 'NOMBRE')
  - índice de bloque 'BLOQUE' (0 = más a la izquierda), consistente con COCO
    (ordenamos anotaciones por x-centrada y enumeramos 0..n-1)

Salida:
  - height_calibrator.joblib  (dict con {'model','features','band_frac','model_type','notes'})
  - calibrator_training_log.csv (qué filas se usaron/omitieron y por qué)

Uso (ejemplo):
python train_height_calibrator.py ^
  --excel "C:\Users\varer\Downloads\CIIS\nuevo_dataset_deep_learning_only.xlsx" ^
  --coco  "C:\Users\varer\Downloads\CIIS\Dataset JSON Etiquetado bloques keypoints.v4i.coco\train\_annotations.coco.json" ^
  --image-roots "C:\Users\varer\Downloads\CIIS\Dataset JSON Etiquetado bloques keypoints.v4i.coco\train" ^
  --outdir "C:\Users\varer\Documents\Proyectogallos\MICAI_FINALE\midas_cal" ^
  --model ridge --midas DPT_Hybrid --device cuda

Requisitos:
  pip install torch torchvision opencv-python numpy pandas joblib openpyxl scikit-learn
  (MiDaS se baja vía torch.hub al primer uso)
"""


import os, sys, json, math, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import joblib
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

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

    # Geo helpers (en píxeles normalizados)
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

# -------------- Utilidades ----------------

def norm_basename(p):
    """Devuelve el basename sin carpetas. Soporta rutas Windows/Unix y posibles '.../images/xxx.jpg'."""
    if pd.isna(p):
        return None
    s = str(p)
    s = s.replace("\\", "/")
    bn = s.split("/")[-1]
    return bn

def find_image_path(basename, image_roots):
    """Busca basename en raíces típicas (root, root/images, root/train/images, root/val/images)."""
    candidates = []
    for r in image_roots:
        r = str(r)
        candidates += [
            os.path.join(r, basename),
            os.path.join(r, "images", basename),
            os.path.join(r, "train", "images", basename),
            os.path.join(r, "val", "images", basename),
        ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def load_coco(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # Mapas útiles
    images = {img["id"]: img for img in data.get("images", [])}
    by_image = {}
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0) == 1: 
            continue
        iid = ann["image_id"]
        by_image.setdefault(iid, []).append(ann)
    return images, by_image

def build_ordered_bboxes_for_image(image_entry, ann_list):
    """Devuelve bboxes ordenados izq→der + su índice (0..n-1) en este orden."""
    # COCO bbox = [x,y,w,h] con x,y esquina sup-izq en píxeles *del archivo*
    rows = []
    for a in ann_list:
        bbox = a.get("bbox", None)
        if not bbox or len(bbox) < 4:
            continue
        x, y, w, h = bbox[:4]
        cx = x + 0.5*w
        rows.append({"bbox": (float(x), float(y), float(w), float(h)), "cx": float(cx)})
    if not rows:
        return []
    rows.sort(key=lambda d: d["cx"])
    for idx, r in enumerate(rows):
        r["block_idx"] = idx
    return rows

# -------------- Entrenamiento ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Ruta al Excel con ground-truth (ALTO PLANTA)")
    ap.add_argument("--coco", required=True, help="Ruta a _annotations.coco.json")
    ap.add_argument("--image-roots", nargs="+", required=True, help="Una o más carpetas raíz donde están las imágenes normales")
    ap.add_argument("--outdir", required=True, help="Carpeta de salida para joblib y logs")
    ap.add_argument("--model", choices=["ridge","rf"], default="ridge", help="Modelo del calibrador")
    ap.add_argument("--midas", choices=["DPT_Hybrid","DPT_Large","MiDaS_small"], default="DPT_Hybrid", help="Tipo de modelo MiDaS")
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--band-frac", type=float, default=0.15, help="Fracción vertical para bandas top/bottom")
    ap.add_argument("--save-depth-cache", action="store_true", help="Guardar mapas de profundidad como .npy para acelerar corridas repetidas")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    depth_cache_dir = outdir / "depth_cache"
    if args.save_depth_cache:
        depth_cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) Cargar Excel
    try:
        df = pd.read_excel(args.excel)
    except Exception as e:
        sys.exit(f"❌ No pude leer el Excel: {e}")

    # Normalizar encabezados y validar columnas clave
    df.columns = [str(c).strip().lower() for c in df.columns]
    col_nombre = "nombre"
    col_bloque = "bloque"
    col_alto   = "alto planta".lower()
    for c in (col_nombre, col_bloque, col_alto):
        if c not in df.columns:
            sys.exit(f"❌ Falta columna en Excel: '{c}' (encabezados: {list(df.columns)})")

    # Extraer basename y casting
    df["basename"] = df[col_nombre].apply(norm_basename)
    df["bloque"]   = pd.to_numeric(df[col_bloque], errors="coerce").astype("Int64")
    df["alto_cm"]  = pd.to_numeric(df[col_alto], errors="coerce")  # se asume en cm (si está en otra unidad, ajusta)

    # Filtrar filas válidas con target
    valid = df.dropna(subset=["basename","bloque","alto_cm"]).copy()

    # 2) Cargar COCO y construir índice por imagen
    images, anns_by_image = load_coco(args.coco)

    # Mapa basename → image_id (COCO) (basado en file_name)
    basename_to_imgid = {}
    for iid, im in images.items():
        bn = norm_basename(im.get("file_name",""))
        if bn:
            basename_to_imgid[bn] = iid

    # 3) Cargar MiDaS
    print(f"[INFO] Cargando MiDaS: {args.midas} (device={args.device}) …")
    midas, midas_tf, device = load_midas(args.midas, args.device)

    # 4) Recorrido de filas y extracción de features
    feat_rows = []
    log_rows  = []

    feature_names = ["h_px","top_med","bot_med","delta","ratio","iqr","cy_n","ybase_n"]

    for idx, row in valid.iterrows():
        bn = row["basename"]
        blk = int(row["bloque"])
        gt_h = float(row["alto_cm"])

        # localizar image_id en COCO
        img_id = basename_to_imgid.get(bn, None)
        if img_id is None:
            log_rows.append({"basename": bn, "bloque": blk, "status": "skip_no_coco"})
            continue

        im_entry = images[img_id]
        ann_list = anns_by_image.get(img_id, [])
        ordered = build_ordered_bboxes_for_image(im_entry, ann_list)
        if not ordered or blk >= len(ordered) or blk < 0:
            log_rows.append({"basename": bn, "bloque": blk, "status": "skip_no_bbox_or_bad_index"})
            continue

        # bbox del BLOQUE (orden izq→der)
        bbox = ordered[blk]["bbox"]  # (x,y,w,h) en px del tamaño original de la imagen

        # abrir imagen
        img_path = find_image_path(bn, args.image_roots)
        if img_path is None:
            log_rows.append({"basename": bn, "bloque": blk, "status": "skip_img_not_found"})
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            log_rows.append({"basename": bn, "bloque": blk, "status": "skip_img_read_error"})
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H_img, W_img = img_rgb.shape[:2]

        # profundidad (cache opcional)
        if args.save_depth_cache:
            cache_file = depth_cache_dir / f"{bn}.npy"
            if cache_file.exists():
                depth = np.load(str(cache_file))
            else:
                depth = run_midas_depth(midas, midas_tf, device, img_rgb)
                np.save(str(cache_file), depth)
        else:
            depth = run_midas_depth(midas, midas_tf, device, img_rgb)

        # features
        feats = depth_feats_from_bbox(depth, bbox, H_img, W_img, band_frac=args.band_frac)
        if feats is None:
            log_rows.append({"basename": bn, "bloque": blk, "status": "skip_bad_roi"})
            continue

        X = [feats[n] for n in feature_names]
        feat_rows.append({
            "basename": bn,
            "bloque": blk,
            "alto_cm": gt_h,
            **{f:fval for f, fval in zip(feature_names, X)}
        })
        log_rows.append({"basename": bn, "bloque": blk, "status": "ok"})

        if (len(feat_rows) % 100) == 0:
            print(f"[INFO] Procesadas {len(feat_rows)} muestras OK…")

    # 5) DataFrame final y entrenamiento
    feats_df = pd.DataFrame(feat_rows)
    log_df   = pd.DataFrame(log_rows)
    log_path = outdir / "calibrator_training_log.csv"
    feats_path = outdir / "calibrator_features_used.csv"
    feats_df.to_csv(feats_path, index=False, encoding="utf-8")
    log_df.to_csv(log_path, index=False, encoding="utf-8")

    if feats_df.empty:
        sys.exit("❌ No se obtuvieron muestras válidas para entrenar el calibrador.")

    X_mat = feats_df[feature_names].to_numpy(dtype=float)
    y_vec = feats_df["alto_cm"].to_numpy(dtype=float)

    print(f"[INFO] Muestras de entrenamiento: {X_mat.shape[0]}  |  Features: {X_mat.shape[1]}")

    if args.model == "ridge":
        # Ridge con búsqueda interna de alpha
        model = RidgeCV(alphas=np.logspace(-3, 3, 13))
        model.fit(X_mat, y_vec)
    else:
        # Random Forest básico
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_mat, y_vec)

    # 6) Guardar calibrador
    calib = {
        "model": model,
        "features": feature_names,
        "band_frac": args.band_frac,
        "model_type": args.model,
        "midas_type": args.midas,
        "notes": "Calibrador de altura (MiDaS→cm) entrenado con Excel 'ALTO PLANTA' y bboxes COCO (orden L→R).",
    }
    out_joblib = outdir / "height_calibrator.joblib"
    joblib.dump(calib, out_joblib)
    print(f"\n✅ Calibrador guardado en: {out_joblib}")

    # 7) Report rápido
    y_pred = model.predict(X_mat)
    mae = np.mean(np.abs(y_pred - y_vec))
    mape = np.mean(np.abs((y_pred - y_vec) / (y_vec + 1e-6))) * 100.0
    print(f"[TRAIN] MAE={mae:.2f} cm | MAPE={mape:.2f}%")
    print(f"[INFO] Log: {log_path}")
    print(f"[INFO] Features usadas: {feats_path}")

if __name__ == "__main__":
    main()
