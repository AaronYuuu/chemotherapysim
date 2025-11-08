# Training Guide - Stratified Cancer Drug Response Prediction

## Overview

This project uses **stratified training** to address treatment line bias:
- **First treatment (N=822)**: Uses XGBoost (better for small datasets)
- **Previous treatment (N=2,112)**: Uses Deep Learning (sufficient data for 23M params)

## Quick Start

### Train Previous Treatment Model (Deep Learning)
```bash
python train.py
```
- Automatically filters for treatment_line >= 2
- Trains 23M parameter cross-attention model
- Saves to: `checkpoints_stratified/previous_treatment/`
- Logs to: `logs/previous_treatment_YYYYMMDD_HHMMSS.log`

### Train First Treatment Model (XGBoost)
```bash
python train_xgboost.py
```
- Automatically filters for treatment_line == 1
- Performs 5-fold cross-validation
- Selects top 500 features (from 9,510)
- Saves to: `checkpoints_stratified/first_treatment_xgboost/`
- AUROC: **0.703** (vs 0.630 for deep learning)

## Directory Structure

```
cancerchemo/
├── train.py                          # Deep learning for previous treatment
├── train_xgboost.py                  # XGBoost for first treatment
├── artifacts/                        # Preprocessed data
│   ├── X_tabular_clean.npy           # Genomic + clinical features
│   ├── X_drug_fp_clean.npy           # Drug fingerprints
│   └── y_clean.npy                   # Target (log PFS)
├── checkpoints_stratified/           # Trained models
│   ├── previous_treatment/           # Deep learning checkpoints
│   │   ├── best_model.pt             # 264MB PyTorch model
│   │   ├── classification_metrics.json
│   │   └── training_curves.png
│   └── first_treatment_xgboost/      # XGBoost checkpoints
│       ├── xgboost_model.json        # 80KB model
│       ├── xgboost_model.pkl
│       ├── selected_features.npy     # 500 feature indices
│       ├── classification_metrics.json
│       └── cv_results.json           # Cross-validation results
└── logs/                             # Training logs (JSON)
    ├── previous_treatment_*.log
    └── xgboost_training_*.log
```

## Performance Summary

| Stratum | Model | N Samples | Parameters | Test AUROC | Status |
|---------|-------|-----------|------------|------------|--------|
| First Treatment | XGBoost | 822 | ~500 trees | **0.703** | ✅ Good |
| Previous Treatment | Deep Learning | 2,112 | 23M | TBD | ⏳ Retrain |

## Why Stratified Training?

### Original Problem
- Single model learned treatment_line as dominant predictor
- SHAP analysis: Only treatment_line and is_combination drove predictions
- All first-line predictions collapsed to 1.5-1.8 months (no drug/genomic effects)

### Solution
1. **Separate models per treatment context** removes treatment line bias
2. **XGBoost for small N** (822 samples, 9,510 features)
   - Feature selection: 9,510 → 500 (eliminates noise)
   - 5-fold CV ensures all samples used for train/val
   - Conservative regularization (max_depth=4, strong L1/L2)
3. **Deep learning for large N** (2,112 samples)
   - Cross-attention captures drug-genomic interactions
   - Sufficient data to train 23M parameters
   - Expected AUROC > 0.70

## Next Steps

1. **Retrain previous treatment model** (if needed):
   ```bash
   python train.py
   ```
   Expected AUROC > 0.70 with N=2,112 samples

2. **Update deployment** (`app.py`):
   - Load XGBoost for treatment_line == 1
   - Load Deep Learning for treatment_line >= 2

3. **Test predictions**:
   - Verify drug/genomic features drive predictions (not just treatment_line)
   - SHAP should show diverse feature importance
   - Predictions should vary with drug selection

## Configuration

### Deep Learning (train.py)
- Dropout: 0.5/0.4/0.5 (genomic/drug/head)
- Weight decay: 1e-3
- Learning rate: 1e-3 with warmup + cosine decay
- Batch size: 32
- Early stopping: patience=15

### XGBoost (train_xgboost.py)
- Max depth: 4 (shallow trees)
- Learning rate: 0.03
- Estimators: 100
- Subsample: 0.8
- L1/L2 regularization: 0.5/1.0
- Feature selection: Top 500 by gain

## References

- Wiens et al. (2014): Patient-level splitting
- Riley et al. (2020): Stratified sampling on outcomes
- Bouthillier et al. (2021): Reproducibility with fixed seeds
- Prechelt (1998): Early stopping best practices
