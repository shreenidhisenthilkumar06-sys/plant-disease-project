# PiTLiD — Plant Leaf Disease Identification

---

## Project Structure

```
PiTLiD_project/
├── model.py               # Inception-V3 + Transfer Learning model
├── data_loader.py         # Data augmentation & loading pipeline
├── cyclical_lr.py         # Cyclical Learning Rate (CLR) callback
├── train.py               # Training script
├── evaluate.py            # Evaluation: metrics, confusion matrix, 10-run test
├── visualize_features.py  # Feature map visualization 
├── prepare_dataset.py     # PlantVillage dataset splitter
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Dataset

Download **PlantVillage** from Kaggle:
```bash
pip install kaggle
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d PlantVillage
```

Then prepare the split (30 images/class for training):
```bash
python prepare_dataset.py --source_dir ./PlantVillage/color --output_dir ./data
```

Expected structure after preparation:
```
data/
├── train/
│   ├── Black_rot/          (30 images)
│   ├── Cedar_apple_rust/   (30 images)
│   ├── Healthy/            (30 images)
│   └── Scab/               (30 images)
├── val/
│   └── ...
└── test/
    └── ...
```

---

## Training

```bash
python train.py \
  --train_dir ./data/train \
  --val_dir   ./data/val \
  --epochs    50 \
  --steps     200 \
  --fine_tune all
```

### Fine-tuning options 
| Argument      | Description                    | Accuracy |
|---------------|--------------------------------|----------|
| `all`         | Fine-tune all layers ✅ BEST   | 99.45%   |
| `freeze_7`    | Freeze first 7 layers          | Lower    |
| `freeze_10`   | Freeze first 10 layers         | Lower    |

### Key hyperparameters 
| Parameter       | Value         |
|-----------------|---------------|
| Optimizer       | RMSprop       |
| Batch size      | 32 (train), 16 (val) |
| Epochs          | 50            |
| Steps           | 200           |
| CLR base LR     | 0.001         |
| CLR max LR      | 0.006         |
| CLR policy      | triangular2   |
| Early stop      | patience=14   |
| Regularization  | L2            |

---

## Evaluation

```bash
python evaluate.py \
  --test_dir   ./data/test \
  --model_path ./saved_model/pitlid_best.h5 \
  --n_runs     10
```

**Results 
| Metric      | Mean    | Std   |
|-------------|---------|-------|
| Accuracy    | 99.45%  | ±0.17 |
| Sensitivity | 99.10%  | ±0.23 |
| Precision   | 98.84%  | ±0.31 |
| F1 Score    | 99.00%  | ±0.23 |

---

## Feature Visualization

```bash
python visualize_features.py \
  --model_path ./saved_model/pitlid_best.h5 \
  --image_path ./sample_leaf.jpg
```

---

## References
- Liu K. & Zhang X. (2023). PiTLiD. *IEEE/ACM TCBB*, 20(2), 1278–1288.
- PlantVillage Dataset: Hughes & Salathé (2015). arXiv:1511.08060
- Inception-V3: Szegedy et al. (2016). CVPR.
- Cyclical LR: Smith (2017). IEEE WACV.
