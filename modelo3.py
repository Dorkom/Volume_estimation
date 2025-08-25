#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baselines tabulares sobre features de bbox.
- Targets: V (m³) directo o H (cm) con cálculo posterior de V usando LARGO y ANCHO del Excel.
- Modelos: RidgeCV y RandomForestRegressor.

Uso típico (Windows):
  python train_tabular_baselines.py ^
    --data-root "C:\\Users\\varer\\Documents\\Proyectogallos\\MICAI_FINALE\\dataset_extra" ^
    --outdir    "C:\\Users\\varer\\Documents\\Proyectogallos\\MICAI_FINALE\\runs_tabular" ^
    --target V

o para altura:
  python train_tabular_baselines.py ^
    --data-root "C:\\Users\\varer\\Documents\\Proyectogallos\\MICAI_FINALE\\dataset_extra" ^
    --outdir    "C:\\Users\\varer\\Documents\\Proyectogallos\\MICAI_FINALE\\runs_tabular" ^
    --target H
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from ast import literal_eval
from typing import Tuple, List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def parse_bbox(b):
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return list(map(float, b))
    if isinstance(b, str):
        try:
            x = literal_eval(b)
            if isinstance(x, (list, tuple)) and len(x) == 4:
                return list(map(float, x))
        except Exception:
            pass
    return [np.nan, np.nan, np.nan, np.nan]

def add_bbox_features(df: pd.DataFrame) -> pd.DataFrame:
    # bboxes vienen como [x, y, w, h] en píxeles
    arr = df["bbox"].apply(parse_bbox).tolist()
    arr = np.array(arr, dtype=float)
    df["bbox_x"] = arr[:,0]
    df["bbox_y"] = arr[:,1]
    df["bbox_w"] = arr[:,2]
    df["bbox_h"] = arr[:,3]

    # derivados
    df["A_px"]   = df["bbox_w"] * df["bbox_h"]
    df["aspect"] = np.divide(df["bbox_w"], df["bbox_h"], out=np.zeros_like(df["bbox_w"]), where=df["bbox_h"]>0)
    df["cx"]     = df["bbox_x"] + 0.5*df["bbox_w"]
    df["cy"]     = df["bbox_y"] + 0.5*df["bbox_h"]
    df["y_base"] = df["bbox_y"] + df["bbox_h"]

    # normalizados por tamaño de imagen original
    # width/height columnas existen en master_* por COCO
    eps = 1e-6
    df["w_norm"]   = df["bbox_w"] / (df["width"]  + eps)
    df["h_norm"]   = df["bbox_h"] / (df["height"] + eps)
    df["cx_norm"]  = df["cx"]     / (df["width"]  + eps)
    df["cy_norm"]  = df["cy"]     / (df["height"] + eps)
    df["ybase_n"]  = df["y_base"] / (df["height"] + eps)
    df["area_n"]   = df["A_px"]   / ((df["width"]*df["height"]) + eps)

    return df

def load_split(data_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(data_root / "train" / "master_train.csv")
    va = pd.read_csv(data_root / "val"   / "master_val.csv")
    return tr, va

def select_features(df: pd.DataFrame) -> List[str]:
    feats = [
        "bbox_w","bbox_h","A_px","aspect",
        "cx","cy","y_base",
        "w_norm","h_norm","cx_norm","cy_norm","ybase_n","area_n",
    ]
    return [f for f in feats if f in df.columns]

def evaluate_volume_metrics(v_true, v_pred) -> Tuple[float, float]:
    v_true = np.asarray(v_true, dtype=float)
    v_pred = np.asarray(v_pred, dtype=float)
    mae = mean_absolute_error(v_true, v_pred)
    mape = np.mean(np.abs((v_true - v_pred) / np.clip(np.abs(v_true), 1e-6, None))) * 100.0
    return mae, mape

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Carpeta dataset_extra con train/ y val/")
    ap.add_argument("--outdir",    required=True, help="Carpeta donde guardar modelos y reportes")
    ap.add_argument("--target",    choices=["V","H"], default="V",
                    help="V: volumen directo (m³). H: altura (cm) y luego V con L&W del Excel")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    data_root = Path(args.data_root)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Cargar splits
    train_df, val_df = load_split(data_root)

    # Filtrar filas con etiquetas disponibles
    if args.target == "V":
        label_col = "M3 PLANTA"
    else:
        label_col = "ALTO PLANTA"
    train_df = train_df[~train_df[label_col].isna()].copy()
    val_df   = val_df[~val_df[label_col].isna()].copy()

    # Features de bbox
    train_df = add_bbox_features(train_df)
    val_df   = add_bbox_features(val_df)

    feature_cols = select_features(train_df)
    if not feature_cols:
        raise RuntimeError("No se generaron columnas de features. Revisa que 'bbox', 'width', 'height' existan en los CSV.")

    # Targets
    y_tr = train_df[label_col].astype(float).values
    y_va = val_df[label_col].astype(float).values

    X_tr = train_df[feature_cols].values
    X_va = val_df[feature_cols].values

    print(f"Train: {X_tr.shape} | Val: {X_va.shape} | target={label_col}")

    # ==========
    # RidgeCV
    # ==========
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5))
    ])
    ridge.fit(X_tr, y_tr)
    pred_va_ridge = ridge.predict(X_va)

    # ==========
    # RandomForest
    # ==========
    rf = RandomForestRegressor(
        n_estimators=600, max_depth=None, min_samples_leaf=3,
        random_state=args.seed, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    pred_va_rf = rf.predict(X_va)

    # ==========
    # Métricas
    # ==========
    if args.target == "V":
        # Directo en m³
        mae_r, mape_r = evaluate_volume_metrics(y_va, pred_va_ridge)
        mae_f, mape_f = evaluate_volume_metrics(y_va, pred_va_rf)
        print("\n=== Baseline Tabular: Target = V (m³) ===")
        print(f"RidgeCV     → MAE={mae_r:.4f} m³ | MAPE={mape_r:.2f}%")
        print(f"RandomForest→ MAE={mae_f:.4f} m³ | MAPE={mape_f:.2f}%")

    else:
        # Target H (cm). Reportamos MAE/MAPE de H y también de V usando L y W del Excel (cm)
        def cm_to_m3(L_cm, W_cm, H_cm):
            return (L_cm * W_cm * H_cm) / 1e6

        # Altura (cm)
        mae_r_h = mean_absolute_error(y_va, pred_va_ridge)
        mape_r_h = np.mean(np.abs((y_va - pred_va_ridge) / np.clip(np.abs(y_va), 1e-6, None))) * 100.0
        mae_f_h = mean_absolute_error(y_va, pred_va_rf)
        mape_f_h = np.mean(np.abs((y_va - pred_va_rf) / np.clip(np.abs(y_va), 1e-6, None))) * 100.0

        # Volumen usando L y W reales del Excel (para aislar el error de altura)
        for col in ["LARGO PLANTA","ANCHO PLANTA"]:
            if col not in val_df.columns:
                raise RuntimeError(f"Falta columna '{col}' en master_val.csv para calcular V.")

        L_cm = val_df["LARGO PLANTA"].astype(float).values
        W_cm = val_df["ANCHO PLANTA"].astype(float).values
        V_true = cm_to_m3(L_cm, W_cm, val_df["ALTO PLANTA"].astype(float).values)

        V_pred_ridge = cm_to_m3(L_cm, W_cm, pred_va_ridge)
        V_pred_rf    = cm_to_m3(L_cm, W_cm, pred_va_rf)

        mae_r_v, mape_r_v = evaluate_volume_metrics(V_true, V_pred_ridge)
        mae_f_v, mape_f_v = evaluate_volume_metrics(V_true, V_pred_rf)

        print("\n=== Baseline Tabular: Target = H (cm) ===")
        print(f"RidgeCV Altura → MAE={mae_r_h:.2f} cm | MAPE={mape_r_h:.2f}%")
        print(f"RF     Altura  → MAE={mae_f_h:.2f} cm | MAPE={mape_f_h:.2f}%")

        print("\n…y Volumen derivado con L y W reales del Excel:")
        print(f"RidgeCV Volumen→ MAE={mae_r_v:.4f} m³ | MAPE={mape_r_v:.2f}%")
        print(f"RF     Volumen → MAE={mae_f_v:.4f} m³ | MAPE={mape_f_v:.2f}%")

    # Guardar modelos
    import joblib
    joblib.dump(ridge, outdir / f"ridge_{args.target}.joblib")
    joblib.dump(rf,    outdir / f"rf_{args.target}.joblib")
    # Guardar lista de features
    (outdir / "features_used.json").write_text(
        json.dumps({"features": feature_cols, "target": args.target}, indent=2), encoding="utf-8"
    )
    print(f"\nModelos guardados en: {outdir}")

if __name__ == "__main__":
    main()
