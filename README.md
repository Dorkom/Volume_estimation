Dataset can be found in the following link:
https://universe.roboflow.com/dorkom/etiquetado-bloques-keypoints/dataset/4
---

````markdown
# Volume Estimation of Travertine Blocks using Keypoint Detection and Homography

This repository contains the full implementation developed for the thesis project **"Volume Estimation of Travertine Blocks Using Keypoint Detection and Homography from Monocular Video"**.  
The system integrates **YOLOv8-Pose**, **MiDaS depth estimation**, and **homography-based scaling** to estimate the real-world dimensions of marble blocks from RGB images.

---

## 🧩 Environment Setup

```bash
pip install ultralytics

# Create and activate virtual environment
python -m venv yolov8env
cd C:\Users\varer\Documents\Keypoints\yolov8env
.\Scripts\activate.ps1

# Upgrade pip and check YOLO installation
python -m pip install --upgrade pip
yolo checks

# (Optional) Reinstall PyTorch with CUDA 11.8 support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
yolo checks
````

---

## 🏋️‍♂️ YOLOv8 Pose Training

To train the model for block keypoint detection:

```bash
yolo pose train \
  model=weights.pt \
  data=C:/Users/varer/Documents/Keypoints/Etiquetado_bloques_keypoints.v4i.yolov8/data.yaml \
  epochs=200 \
  patience=27 \
  imgsz=960 \
  batch=8 \
  name=yolov8m-marmol-finetune \
  device=0
```

---

## 📁 Repository Structure

| File                           | Description                                                                                                                                                |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **calibracion_midas.py**       | Calibrates a regression model that maps MiDaS depth features to real height in centimeters, using reference measurements.                                  |
| **resultados_midas_altura.py** | Predicts the **height of each detected block** using the MiDaS model and the saved `height_calibrator.joblib`. Also draws overlays with estimated heights. |
| **resultados_extra.py**        | Generates complementary metrics and summary results for validation or visualization of model outputs.                                                      |
| **modelo2.py**                 | Experimental script for training or testing an alternative height/volume regression model.                                                                 |
| **modelo3.py**                 | Refined version of `modelo2.py`, possibly including extended feature sets or validation improvements.                                                      |
| **recorte.py**                 | Handles image cropping or region extraction, preparing inputs for training or evaluation.                                                                  |
| **roboflowapi.py**             | Automates image and annotation retrieval from the **Roboflow API**, used for dataset synchronization.                                                      |
| **final.py**                   | Main pipeline integrating **detection**, **depth estimation**, and **volume computation** using calibrated parameters.                                     |
| **complete_table.png**         | Visual summary of model performance or experimental results (e.g., error table or accuracy comparison).                                                    |
| **pipeline.png**               | Diagram of the complete vision pipeline: YOLOv8 Pose detection → MiDaS depth → homography scaling → volume estimation.                                     |

---

## 🔍 Execution Example

To predict block heights from an image and its Roboflow JSON output:

```bash
python resultados_midas_altura.py ^
  --img  "C:\ruta\frame.jpg" ^
  --json "C:\ruta\resultado.json" ^
  --calib "C:\...\height_calibrator.joblib" ^
  --device cuda --draw out.png
```

**Output example:**

```
[INFO] MiDaS=DPT_Hybrid  device=cuda

Bloques detectados: 3 (izq→der)

[Bloque 0]  Alto(MiDaS-cal) ≈ 87.2 cm
[Bloque 1]  Alto(MiDaS-cal) ≈ 83.6 cm
[Bloque 2]  Alto(MiDaS-cal) ≈ 91.4 cm

🖼️ Overlay guardado en: out.png
```

---

## 🧠 Core Concepts

* **MiDaS Depth Estimation:** Extracts relative depth maps from monocular RGB images.
* **Feature Calibration:** Uses median depth differences and pixel height ratios to predict real height in cm.
* **Homography Scaling:** Converts image-based coordinates to metric dimensions using planar references.
* **YOLOv8-Pose:** Detects keypoints and bounding boxes of marble blocks for geometric analysis.

---
