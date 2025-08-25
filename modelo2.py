#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrena un modelo CNN para predecir V^(1/3) (m) a partir de crops de bloques.

- Lee los CSVs generados por recorte.py (train/val).
- Busca los crops en train/crops y val/crops.
- Modelo: ResNet-18 preentrenada → cabeza de regresión (1 salida).
- Pérdida: SmoothL1Loss (Huber) sobre V^(1/3).
- Métricas: MAE y MAPE sobre V (m^3), reportadas en validación.
- Guarda el mejor checkpoint (menor MAE en m^3).

Uso (ajusta --data-root a tu carpeta dataset_extra):

  python train_volume_cuberoot.py ^
    --data-root "C:\\Users\\varer\\Documents\\Proyectogallos\\MICAI_FINALE\\dataset_extra" ^
    --epochs 40 --batch-size 32 --lr 1e-4 --num-workers 4 ^
    --outdir "C:\\Users\\varer\\Documents\\Proyectogallos\\MICAI_FINALE\\runs_ai"

"""

import os
import math
import time
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Dataset
# -------------------------
class BlocksVolumeDataset(Dataset):
    """
    Lee master_train.csv / master_val.csv y arma ruta al crop:
      crop_name = f"{Path(file_name).stem}__ann{id}.jpg"
      crop_path = <split>/crops/crop_name
    Target: (M3 PLANTA)^(1/3)  (en metros)
      * Si M3 está en m^3, la raíz cúbica también queda en metros.
    """
    def __init__(self, csv_path: str, crops_dir: str, transform=None, drop_na=True):
        self.df = pd.read_csv(csv_path)
        # Filtramos filas sin M3
        if drop_na and "M3 PLANTA" in self.df.columns:
            self.df = self.df[~self.df["M3 PLANTA"].isna()].copy()
        # Asegurar columnas necesarias
        required = ["file_name", "id", "M3 PLANTA"]
        for r in required:
            if r not in self.df.columns:
                raise ValueError(f"Falta columna requerida en {csv_path}: '{r}'")
        self.crops_dir = Path(crops_dir)
        self.transform = transform

        # Construir rutas a crops
        stems = self.df["file_name"].astype(str).apply(lambda s: Path(s).stem)
        ann_ids = self.df["id"].astype(int)
        crop_names = [f"{stem}__ann{ann_id}.jpg" for stem, ann_id in zip(stems, ann_ids)]
        self.df["crop_path"] = [str(self.crops_dir / cn) for cn in crop_names]

        # Raíz cúbica del volumen como target
        self.df["target_cuberoot"] = np.cbrt(self.df["M3 PLANTA"].astype(float))

        # Verificación de existencia (opcional)
        missing = (~self.df["crop_path"].apply(os.path.exists)).sum()
        if missing > 0:
            print(f"[WARN] {missing} crops no encontrados en {self.crops_dir} (se ignorarán).")
            self.df = self.df[self.df["crop_path"].apply(os.path.exists)].copy()

        self.df.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["crop_path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y_cuberoot = float(row["target_cuberoot"])
        y_vol = float(row["M3 PLANTA"])
        return img, y_cuberoot, y_vol, img_path


# -------------------------
# Modelo
# -------------------------
class ResNetRegressor(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True, out_features=1):
        super().__init__()
        # Cargar backbone
        if backbone_name == "resnet18":
            self.backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            in_feats = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name == "resnet50":
            self.backbone = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            in_feats = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("Backbone no soportado. Usa 'resnet18' o 'resnet50'.")

        # Cabeza de regresión
        self.head = nn.Sequential(
            nn.Linear(in_feats, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, out_features),  # 1 salida: V^(1/3)
        )

    def forward(self, x):
        z = self.backbone(x)
        y = self.head(z)
        return y.squeeze(1)


# -------------------------
# Entrenamiento / Evaluación
# -------------------------
def mae(y_true, y_pred):
    return float(torch.mean(torch.abs(y_true - y_pred)).item())

def mape(y_true, y_pred, eps=1e-6):
    denom = torch.clamp(torch.abs(y_true), min=eps)
    return float(torch.mean(torch.abs((y_true - y_pred) / denom)).item() * 100.0)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.SmoothL1Loss(beta=0.1)  # Huber
    losses = []
    all_v_true = []
    all_v_pred = []

    for imgs, y_cbr, y_v, _ in loader:
        imgs = imgs.to(device)
        y_cbr = y_cbr.to(device)
        y_v = y_v.to(device)

        yhat_cbr = model(imgs)                     # predicción de V^(1/3)
        loss = loss_fn(yhat_cbr, y_cbr)
        losses.append(loss.item())

        # Convertimos a V (m^3) para métricas en volumen
        yhat_v = torch.clamp(yhat_cbr, min=0.0) ** 3
        all_v_true.append(y_v)
        all_v_pred.append(yhat_v)

    v_true = torch.cat(all_v_true)
    v_pred = torch.cat(all_v_pred)

    val_mae = mae(v_true, v_pred)
    val_mape = mape(v_true, v_pred)
    return np.mean(losses), val_mae, val_mape


def train(args):
    set_seed(args.seed)
    device = default_device()
    print(f"Device: {device}")

    data_root = Path(args.data_root)
    train_csv = data_root / "train" / "master_train.csv"
    val_csv   = data_root / "val"   / "master_val.csv"
    train_crops = data_root / "train" / "crops"
    val_crops   = data_root / "val"   / "crops"

    # Transforms
    train_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        # Augmentación leve; evita cambios geométricos fuertes
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Datasets y loaders
    train_ds = BlocksVolumeDataset(str(train_csv), str(train_crops), transform=train_tfms)
    val_ds   = BlocksVolumeDataset(str(val_csv),   str(val_crops),   transform=val_tfms)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Modelo
    model = ResNetRegressor(backbone_name=args.backbone, pretrained=True, out_features=1).to(device)

    # Optimizador y scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=3, verbose=True)

    loss_fn = nn.SmoothL1Loss(beta=0.1)  # Huber sobre V^(1/3)

    # Salidas
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    best_mae = float("inf")
    history = []

    # Loop de entrenamiento
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for imgs, y_cbr, y_v, _ in train_loader:
            imgs = imgs.to(device)
            y_cbr = y_cbr.to(device)

            optimizer.zero_grad(set_to_none=True)
            yhat_cbr = model(imgs)
            loss = loss_fn(yhat_cbr, y_cbr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        # Validación
        val_loss, val_mae, val_mape = evaluate(model, val_loader, device)
        scheduler.step(val_mae)  # reducimos LR cuando MAE (en m³) se estanca

        dt = time.time() - t0
        tr_loss = float(np.mean(epoch_losses))
        print(f"[Epoch {epoch:03d}] "
              f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} "
              f"| val_MAE_m3={val_mae:.4f} | val_MAPE%={val_mape:.2f} | time={dt:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_mae_m3": val_mae,
            "val_mape_pct": val_mape,
            "time_sec": dt
        })

        # Guardar mejor modelo por MAE(m^3)
        if val_mae < best_mae:
            best_mae = val_mae
            best_path = outdir / "best_resnet18_cuberoot.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_mae_m3": best_mae,
                "args": vars(args)
            }, best_path)
            print(f"  ↳ Nuevo mejor modelo guardado en: {best_path}")

    # Guardar historia
    hist_path = outdir / "training_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"Historial guardado en: {hist_path}")


# -------------------------
# Main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Carpeta dataset_extra (contiene train/ y val/)")
    ap.add_argument("--outdir", required=True, help="Carpeta donde guardar checkpoints y logs")
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18","resnet50"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
