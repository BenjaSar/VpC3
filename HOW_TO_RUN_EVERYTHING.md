# 🚀 Complete Guide: How to Run Everything

## 📋 Quick Summary

This project has several modules that need to be run in sequence. Here's the complete workflow:

```
1. Convert Dataset (SVG → PNG)
2. Run EDA Analysis
3. Run Preprocessing
4. Test Dataset/Dataloaders
5. Start Training
```

---

## 🎯 Step-by-Step Instructions

### Step 1: Convert CubiCasa5K Dataset

**Purpose:** Convert SVG annotations to PNG masks

**Command:**
```bash
# First, install the converter
pip install cairosvg

# Test with 10 samples
python data/svg_to_png_converter.py --test 10

# If successful, convert all 992 floor plans
python data/svg_to_png_converter.py
```

**Input:** `data/cubicasa5k/cubicasa5k/high_quality/` (992 floor plans with SVG annotations)

**Output:** `data/cubicasa5k_converted/` (images + PNG annotations)

**Time:** ~15-30 minutes for full conversion

---

### Step 2: Run EDA (Exploratory Data Analysis)

**Purpose:** Analyze dataset quality and generate reports

**Command:**
```bash
python src/eda/eda_analysis.py --dataset_path data/cubicasa5k_converted --dataset_type cubicasa5k --output_dir eda_output
```

**Input:** `data/cubicasa5k_converted/`

**Output:** 
- `eda_output/EDA_REPORT.txt` - Human-readable report
- `eda_output/eda_report.json` - Machine-readable data
- `eda_output/*.png` - 7 visualization plots

**Time:** ~5-10 minutes

---

### Step 3: Run Preprocessing

**Purpose:** Resize images and validate dataset

**Command:**
```bash
python run_preprocessing.py
```

**What it does:**
1. Validates dataset (checks all images have matching annotations)
2. Resizes all images to 512x512
3. Saves preprocessed data

**Input:** `data/cubicasa5k_converted/`

**Output:** `data/processed/` (resized images and masks)

**Time:** ~5-10 minutes

**You can edit the configuration in the script:**
```python
# Open run_preprocessing.py and modify:
INPUT_DIR = Path("data/cubicasa5k_converted")
OUTPUT_DIR = Path("data/processed")
TARGET_SIZE = 512  # Change this for different size
```

---

### Step 4: Test Dataset & Dataloaders

**Purpose:** Test PyTorch dataloaders and visualize samples

**Command:**
```bash
python run_dataset.py
```

**What it does:**
1. Creates train/val/test dataloaders
2. Tests loading a batch
3. Checks memory usage
4. Measures loading speed
5. Generates visualization

**Input:** `data/processed/`

**Output:** 
- Console logs with statistics
- `dataloader_visualization.png` - Sample images and masks

**Time:** <1 minute

---

### Step 5: Start Training (Coming Soon)

**Command:**
```bash
python train.py  # You'll need to create this
```

**How to use dataloaders in training:**
```python
from data.dataset import create_dataloaders

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    images_dir="data/processed/images",
    masks_dir="data/processed/annotations",
    batch_size=8,
    num_workers=4,
    image_size=512,
    num_classes=34
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        images = batch['image'].cuda()
        masks = batch['mask'].cuda()
        
        # Your training code
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
```

---

## 📊 Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Raw CubiCasa5K Dataset                                      │
│ data/cubicasa5k/cubicasa5k/high_quality/                   │
│ (992 floor plans with SVG annotations)                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ python data/svg_to_png_converter.py
                  │ (~15-30 min)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ Converted Dataset                                            │
│ data/cubicasa5k_converted/                                   │
│ ├── images/ (992 PNG images)                                │
│ └── annotations/ (992 PNG masks)                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ├─────────────────────────┐
                  │                         │
                  │ EDA (optional)          │ Preprocessing (required)
                  │                         │
                  ▼                         ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│ EDA Reports              │  │ Preprocessed Dataset          │
│ eda_output/              │  │ data/processed/               │
│ ├── reports              │  │ ├── images/ (512x512)        │
│ └── visualizations       │  │ └── annotations/ (512x512)   │
└──────────────────────────┘  └────────────┬─────────────────┘
                                            │
                                            │ python run_dataset.py
                                            │ (~1 min)
                                            ▼
                              ┌──────────────────────────────┐
                              │ PyTorch DataLoaders Ready    │
                              │ - Train Loader               │
                              │ - Val Loader                 │
                              │ - Test Loader                │
                              └────────────┬─────────────────┘
                                           │
                                           │ python train.py
                                           │
                                           ▼
                              ┌──────────────────────────────┐
                              │ Training                     │
                              │ models/checkpoints/          │
                              └──────────────────────────────┘
```

---

## 🔧 Troubleshooting

### Problem: "Module not found"
```bash
# Make sure you're in the project root directory
cd F:\IA\VpCIII\floorplan-vit-classifier

# Try running again
python run_preprocessing.py
```

### Problem: "Dataset not found"
```bash
# Check which step you're on
# Did you convert the dataset?
ls data/cubicasa5k_converted/images/  # Should show files

# Did you preprocess?
ls data/processed/images/  # Should show files
```

### Problem: "Out of memory"
Edit the batch size in `run_dataset.py`:
```python
CONFIG = {
    'batch_size': 4,  # Reduce from 8 to 4
    # ...
}
```

### Problem: "Too slow"
For Windows, set `num_workers=0` in `run_dataset.py`:
```python
CONFIG = {
    'num_workers': 0,  # Use 0 for Windows
    # ...
}
```

---

## 📝 File Reference

### Executable Scripts (Run Directly)

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `data/svg_to_png_converter.py` | Convert SVG to PNG | CubiCasa5K raw | Converted PNG |
| `src/eda/eda_analysis.py` | Dataset analysis | Converted PNG | EDA reports |
| `run_preprocessing.py` | Preprocess data | Converted PNG | Processed 512x512 |
| `run_dataset.py` | Test dataloaders | Processed data | Visualization |

### Utility Modules (Import Only)

| Module | Purpose | How to Use |
|--------|---------|------------|
| `src/preprocessing.py` | Image utilities | Import in scripts |
| `data/dataset.py` | PyTorch Dataset | Import in training |
| `src/utils/logging_config.py` | Logging setup | Import for logs |

---

## 🎓 Understanding the Modules

### `src/preprocessing.py` - UTILITY MODULE
**DO NOT RUN DIRECTLY**

This provides classes you import:
```python
from src.preprocessing import ImagePreprocessor, DataValidator

# Use in your code
preprocessor = ImagePreprocessor()
img_resized = preprocessor.resize_image(img, 512)
```

### `data/dataset.py` - PYTORCH DATASET
**DO NOT RUN DIRECTLY**

This provides PyTorch Dataset class:
```python
from data.dataset import create_dataloaders

# Use in training
train_loader, val_loader, test_loader = create_dataloaders(...)
```

### `run_preprocessing.py` - EXECUTABLE SCRIPT
**RUN THIS DIRECTLY**

Combines preprocessing utilities into a complete pipeline:
```bash
python run_preprocessing.py
```

### `run_dataset.py` - EXECUTABLE SCRIPT
**RUN THIS DIRECTLY**

Tests dataloaders and creates visualizations:
```bash
python run_dataset.py
```

---

## 🚀 Quick Start (TL;DR)

```bash
# Complete workflow in 5 commands:

# 1. Install converter
pip install cairosvg

# 2. Convert dataset (30 min)
python data/svg_to_png_converter.py

# 3. Run EDA (10 min)
python src/eda/eda_analysis.py --dataset_path data/cubicasa5k_converted

# 4. Preprocess (10 min)
python run_preprocessing.py

# 5. Test dataloaders (1 min)
python run_dataset.py

# 6. Ready for training!
```

**Total Time:** ~50 minutes to production-ready dataset

---

## 📚 Documentation Files

| File | Content |
|------|---------|
| `CUBICASA5K_CONVERSION_GUIDE.md` | Detailed conversion instructions |
| `DATASET_DOWNLOAD_GUIDE.md` | Dataset sources and downloads |
| `PREPROCESSING_USAGE_GUIDE.md` | Preprocessing module details |
| `HOW_TO_RUN_EVERYTHING.md` | **This file - Complete workflow** |

---

## ✅ Checklist

Use this checklist to track your progress:

- [ ] Installed cairosvg (`pip install cairosvg`)
- [ ] Converted CubiCasa5K dataset (`python data/svg_to_png_converter.py`)
- [ ] Ran EDA analysis (`python src/eda/eda_analysis.py ...`)
- [ ] Ran preprocessing (`python run_preprocessing.py`)
- [ ] Tested dataloaders (`python run_dataset.py`)
- [ ] Reviewed `dataloader_visualization.png`
- [ ] Ready to start training!

---

## 🎯 Next Steps

After completing all steps above:

1. **Review Generated Files:**
   - EDA reports in `eda_output/`
   - Processed data in `data/processed/`
   - Dataloader visualization

2. **Implement Training:**
   - Create `train.py`
   - Use dataloaders from `data/dataset.py`
   - Save checkpoints to `models/checkpoints/`

3. **Monitor Training:**
   - Use MLflow for experiment tracking
   - Log metrics and losses
   - Save best models

---

**Last Updated:** October 27, 2025  
**Project:** floorplan-vit-classifier
