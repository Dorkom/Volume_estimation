#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO -> Crops + CSV + Split Train/Val por root_name (robusto a encabezados Excel)

Uso típico (Windows):
  python recorte.py ^
    --coco "C:\...\_annotations.coco.json" ^
    --excel "C:\...\nuevo_dataset_deep_learning_only.xlsx" ^
    --image-roots "C:\...\train\images" ^
    --out-dir "C:\...\dataset_extra" ^
    --val-ratio 0.2 ^
    --stratify-bins 8 ^
    --seed 42 ^
    --pad 0.12 ^
    --save-crops

Opcional: forzar nombre de columna del Excel con --name-col
  --name-col NOMBRE
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

try:
    from unidecode import unidecode
except Exception:
    # fallback si no está instalado
    def unidecode(x): 
        return str(x)

# ---------------------------
# Utilidades de nombres
# ---------------------------
def basename_only(path_str: str) -> str:
    return os.path.basename(str(path_str)).replace("\\", "/")

def to_root_name(fn: str) -> str:
    base = basename_only(fn)
    if ".rf." in base:
        left, _ = base.split(".rf.", 1)
        ext = os.path.splitext(base)[1] or ".jpg"
        if "." not in left:
            return left + ext
        else:
            left_stem, _ = os.path.splitext(left)
            return left_stem + ext
    return base

def find_image_fullpath(file_name: str, search_roots: List[str]) -> str:
    if os.path.isabs(file_name) and os.path.exists(file_name):
        return file_name
    if os.path.exists(file_name):
        return os.path.abspath(file_name)
    for root in search_roots:
        candidate = os.path.join(root, file_name)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    subs = ["images", "train/images", "valid/images", "test/images", "train", "valid", "test"]
    base = basename_only(file_name)
    for root in search_roots:
        for sub in subs:
            candidate = os.path.join(root, sub, base)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
    return file_name

# ---------------------------
# COCO helpers
# ---------------------------
def load_coco(json_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images_df = pd.DataFrame(coco.get("images", []))
    anns_df   = pd.DataFrame(coco.get("annotations", []))
    cats_df   = pd.DataFrame(coco.get("categories", []))
    if "file_name" not in images_df.columns:
        raise ValueError("El COCO no tiene 'file_name' en 'images'.")
    if "bbox" not in anns_df.columns:
        raise ValueError("El COCO no tiene 'bbox' en 'annotations'.")
    if "keypoints" not in anns_df.columns:
        anns_df["keypoints"] = [[]]*len(anns_df)
    return images_df, anns_df, cats_df

def compute_block_index_per_image(anns_df: pd.DataFrame) -> pd.DataFrame:
    df = anns_df.copy()
    def _xc(b):
        try:
            return float(b[0]) + float(b[2]) / 2.0
        except Exception:
            return np.nan
    df["x_center"] = df["bbox"].apply(_xc)
    df = df.sort_values(["image_id", "x_center"], ascending=[True, True])
    df["BLOQUE"] = df.groupby("image_id").cumcount()
    return df

def safe_crop(img_path: str, bbox: List[float], pad_ratio: float = 0.12) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    x, y, w, h = bbox
    pad_w = w * pad_ratio
    pad_h = h * pad_ratio
    x0 = max(0, int(np.floor(x - pad_w)))
    y0 = max(0, int(np.floor(y - pad_h)))
    x1 = min(W, int(np.ceil(x + w + pad_w)))
    y1 = min(H, int(np.ceil(y + h + pad_h)))
    return img.crop((x0, y0, x1, y1))

# ---------------------------
# Split helpers
# ---------------------------
def stratify_bins_from_series(y: pd.Series, n_bins: int) -> pd.Series:
    y_valid = y.dropna()
    if len(y_valid) < n_bins:
        return pd.Series(index=y.index, data=0, dtype=int)
    try:
        bins = pd.qcut(y, q=n_bins, duplicates="drop", labels=False)
    except Exception:
        bins = pd.Series(index=y.index, data=0, dtype=int)
    return bins.astype(int)

def train_val_split_by_root(master_df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42, stratify_bins: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    tmp = (master_df.groupby("root_name", dropna=False)["M3 PLANTA"]
           .mean().rename("M3_mean"))
    root_df = tmp.reset_index()
    if stratify_bins and "M3_mean" in root_df.columns:
        root_df["strat"] = stratify_bins_from_series(root_df["M3_mean"], stratify_bins)
    else:
        root_df["strat"] = 0
    val_indices = []
    for _, group in root_df.groupby("strat"):
        idx = group.index.tolist()
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio))) if len(idx) > 1 else 1
        chosen = [root_df.loc[i, "root_name"] for i in idx[:n_val]]
        val_indices.extend(chosen)
    val_roots = set(val_indices)
    train_roots = set(root_df["root_name"]) - val_roots
    train_df = master_df[master_df["root_name"].isin(train_roots)].copy()
    val_full = master_df[master_df["root_name"].isin(val_roots)].copy()
    # dedup validación
    # ---- Asegurar que 'basename' existe en val_full ----
    if "basename" not in val_full.columns:
        val_full["basename"] = val_full["file_name"].apply(lambda s: os.path.basename(str(s)) if pd.notna(s) else np.nan)

    val_df = (val_full.sort_values(["root_name", "basename"])
                     .drop_duplicates(subset=["root_name", "BLOQUE", "id"], keep="first"))
    return train_df, val_df

# ---------------------------
# Normalización encabezados Excel
# ---------------------------
def norm_col(s: str) -> str:
    return unidecode(str(s)).strip().lower().replace("\u00a0"," ")

def find_name_col(excel_df: pd.DataFrame, forced: Optional[str] = None) -> str:
    original_cols = list(excel_df.columns)
    norm_map = {c: norm_col(c) for c in original_cols}
    excel_df.rename(columns=norm_map, inplace=True)
    if forced:
        forced_norm = norm_col(forced)
        if forced_norm in excel_df.columns:
            print(f"[INFO] Usando columna de nombres forzada: '{forced}' -> '{forced_norm}'")
            return forced_norm
        else:
            raise ValueError(f"--name-col '{forced}' no existe. Columnas normalizadas: {list(excel_df.columns)}")
    candidates = ("nombre","name","file_name","filename","archivo","ruta","path")
    for cand in candidates:
        if cand in excel_df.columns:
            print(f"[INFO] Columna de nombres detectada: '{cand}'")
            return cand
    raise ValueError(f"No encuentro columna de nombre de imagen en el Excel. Columnas disponibles: {list(excel_df.columns)}")

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="COCO -> Crops + CSV + Split Train/Val por root_name (robusto)")
    ap.add_argument("--coco", required=True, help="Ruta a _annotations.coco.json")
    ap.add_argument("--excel", required=True, help="Ruta a Excel (incluye NOMBRE y BLOQUE)")
    ap.add_argument("--name-col", default=None, help="Forzar el nombre de la columna con la ruta/nombre de archivo (ej. 'NOMBRE')")
    ap.add_argument("--image-roots", nargs="*", default=[], help="Carpetas donde buscar imágenes reales")
    ap.add_argument("--out-dir", required=True, help="Directorio de salida (se crearán subcarpetas train/val)")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Proporción para validación (por root_name)")
    ap.add_argument("--stratify-bins", type=int, default=None, help="N° de bins para estratificar por M3 (opcional)")
    ap.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para el split")
    ap.add_argument("--pad", type=float, default=0.12, help="Padding relativo para los recortes")
    ap.add_argument("--save-crops", action="store_true", help="Si se pasa, guarda los crops en train/ y val/")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)

    # 1) COCO + índice de BLOQUE
    images_df, anns_df, cats_df = load_coco(args.coco)
    img_meta = images_df.set_index("id")[["file_name", "width", "height"]]
    anns_df = anns_df.join(img_meta, on="image_id", how="left")
    anns_df["basename"] = anns_df["file_name"].apply(basename_only)
    anns_df = compute_block_index_per_image(anns_df)
    anns_df["img_abspath"] = anns_df["file_name"].apply(lambda fn: find_image_fullpath(fn, args.image_roots))
    anns_df["img_exists"] = anns_df["img_abspath"].apply(os.path.exists)
    anns_df["root_name"] = anns_df["basename"].apply(to_root_name)

    # 2) Excel robusto a encabezados
    excel_df = pd.read_excel(args.excel)
    print("[DEBUG] Columnas originales del Excel:", list(excel_df.columns))
    name_col = find_name_col(excel_df, forced=args.name_col)

    # BLOQUE
    if "bloque" not in excel_df.columns:
        raise ValueError(f"No encuentro columna 'BLOQUE' (normalizada a 'bloque'). Columnas: {list(excel_df.columns)}")
    excel_df["bloque"] = excel_df["bloque"].astype(int)

    # Generar basename y root_name desde el nombre/ruta en Excel
    excel_df["basename"] = excel_df[name_col].astype(str).apply(basename_only)
    excel_df["root_name"] = excel_df["basename"].apply(to_root_name)

    # Renombrar medidas si existen
    # (quedan como estaban si ya venían así; solo mapeamos equivalentes comunes)
    col_map_candidates = {
        "largo planta":"LARGO PLANTA", "ancho planta":"ANCHO PLANTA", "alto planta":"ALTO PLANTA",
        "m3 planta":"M3 PLANTA", "m3":"M3 PLANTA", "volumen":"M3 PLANTA"
    }
    for k,v in col_map_candidates.items():
        if k in excel_df.columns and v not in excel_df.columns:
            excel_df.rename(columns={k: v}, inplace=True)

    # 3) Join por (root_name, BLOQUE)
    merged = pd.merge(
        anns_df.rename(columns={"root_name":"root_name_coco", "BLOQUE":"BLOQUE"}),
        excel_df.rename(columns={"root_name":"root_name_xls", "bloque":"BLOQUE"}),
        left_on=["root_name_coco", "BLOQUE"],
        right_on=["root_name_xls", "BLOQUE"],
        how="left",
    )

    # 4) Master CSV
    cols_keep = [
        "image_id","id","file_name","img_abspath","img_exists",
        "basename","root_name_coco","BLOQUE","bbox","width","height",
        "LARGO PLANTA","ANCHO PLANTA","ALTO PLANTA","M3 PLANTA",
    ]
    cols_keep = [c for c in cols_keep if c in merged.columns]
    master_df = merged[cols_keep].copy()
    master_df.rename(columns={"root_name_coco":"root_name"}, inplace=True)
# ---- Asegurar que 'basename' existe en master_df ----
    if "basename" not in master_df.columns:
        master_df["basename"] = master_df["file_name"].apply(lambda s: os.path.basename(str(s)) if pd.notna(s) else np.nan)


    master_csv = out_dir / "master_blocks_for_DL.csv"
    master_df.to_csv(master_csv, index=False, encoding="utf-8")

    # 5) Split por root_name
    if "M3 PLANTA" not in master_df.columns:
        print("[WARN] No se encontró 'M3 PLANTA'; split sin estratificación.")
        strat_bins = None
    else:
        strat_bins = args.stratify_bins

    train_df, val_df = train_val_split_by_root(master_df, val_ratio=args.val_ratio, seed=args.seed, stratify_bins=strat_bins)

    train_csv = out_dir / "train" / "master_train.csv"
    val_csv   = out_dir / "val"   / "master_val.csv"
    train_df.to_csv(train_csv, index=False, encoding="utf-8")
    val_df.to_csv(val_csv, index=False, encoding="utf-8")

    # 6) Guardar crops (opcional)
    if args.save_crops:
        train_crops_dir = out_dir / "train" / "crops"
        val_crops_dir   = out_dir / "val"   / "crops"
        train_crops_dir.mkdir(parents=True, exist_ok=True)
        val_crops_dir.mkdir(parents=True, exist_ok=True)

        def _save_split(df: pd.DataFrame, out_crops_dir: Path, pad: float) -> Tuple[int,int]:
            saved = 0
            skipped = 0
            for _, row in df.iterrows():
                if not row.get("img_exists", False):
                    skipped += 1
                    continue
                bbox = row.get("bbox", None)
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    skipped += 1
                    continue
                try:
                    crop = safe_crop(row["img_abspath"], bbox, pad_ratio=float(pad))
                    base = os.path.splitext(os.path.basename(str(row.get("file_name","unknown.jpg"))))[0]
                    crop_name = f"{base}__ann{int(row['id'])}.jpg"
                    out_path = out_crops_dir / crop_name
                    crop.save(out_path, quality=95)
                    saved += 1
                except Exception:
                    skipped += 1
            return saved, skipped

        tr_saved, tr_skipped = _save_split(train_df, train_crops_dir, args.pad)
        va_saved, va_skipped = _save_split(val_df,   val_crops_dir,   args.pad)
        print(f"[Crops] Train: guardados={tr_saved}, omitidos={tr_skipped}  → {train_crops_dir}")
        print(f"[Crops] Val  : guardados={va_saved}, omitidos={va_skipped}  → {val_crops_dir}")

    # 7) Resumen
    total_anns = len(master_df)
    uniq_images = master_df["basename"].nunique()
    img_found = int(master_df["img_exists"].sum()) if "img_exists" in master_df.columns else None
    m3_missing = int(master_df["M3 PLANTA"].isna().sum()) if "M3 PLANTA" in master_df.columns else None

    print("\n=== RESUMEN ===")
    print(f"Anotaciones (bloques total): {total_anns}")
    print(f"Imágenes únicas (por basename): {uniq_images}")
    if img_found is not None:
        print(f"Imágenes encontradas: {img_found} | no encontradas: {total_anns - img_found}")
    if m3_missing is not None:
        print(f"Filas sin M3 (tras join): {m3_missing}")
    print(f"CSV maestro: {master_csv}")
    print(f"Train CSV : {train_csv}")
    print(f"Val   CSV : {val_csv}")
    if args.save_crops:
        print("Crops guardados en subcarpetas train/crops y val/crops.")
    print("================")

if __name__ == "__main__":
    main()
