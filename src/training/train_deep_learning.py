"""
Training script for cancer drug response prediction model.

This script trains ONLY on previous treatment data (treatment_line >= 2)
for deep learning. First treatment uses XGBoost (see train_xgboost.py).

Evidence-based design:
- PATIENT-LEVEL split: 70/15/15 (Wiens et al. 2014, prevents data leakage)
- Stratified sampling on patient median PFS (Riley et al. 2020)
- Reproducible splits with fixed random seed (Bouthillier et al. 2021)
- Validation-based early stopping (Prechelt 1998, Neural Networks)
- Learning rate warmup + cosine decay (Goyal et al. 2017, Loshchilov & Hutter 2017)

CRITICAL: All treatments from the same patient stay in the same split to prevent
the model from "memorizing" patient-specific characteristics. This ensures the
model generalizes to NEW PATIENTS in clinical practice.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Import model from src.models
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ImprovedDrugResponseModel, create_optimizer, create_scheduler

# ===================================================================
# CONFIGURATION
# ===================================================================

# Set random seeds for reproducibility (Bouthillier et al. 2021)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Paths (relative to project root)
DATA_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_stratified" / "previous_treatment"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Training hyperparameters
CONFIG = {
    # Data split (Collins et al. 2015 - 70/15/15 for N<5000)
    "train_size": 0.70,
    "val_size": 0.15,
    "test_size": 0.15,
    
    # Model architecture
    "genomic_dim": 1318,  # Updated for comprehensive clinical features (was 1240)
    "drug_fp_dim": 8192,
    "embed_dim": 256,
    "dropout_genomic": 0.5,    # Increased from 0.4 to reduce overfitting
    "dropout_drug": 0.4,       # Increased from 0.3 to reduce overfitting
    "dropout_head": 0.5,       # Increased from 0.4 to reduce overfitting
    
    # Training
    "batch_size": 32,          # Way et al. 2018 - 32-64 for genomic data
    "num_epochs": 100,         # Increased - model was still learning at epoch 10
    "learning_rate": 1e-3,     # AdamW default
    "weight_decay": 1e-3,      # Increased from 1e-4 to reduce overfitting
    "warmup_epochs": 10,       # 10% of total epochs for gradual warmup
    
    # Early stopping
    "patience": 15,            # Reduced from 20 - stop sooner if overfitting
    "min_delta": 1e-4,         # Minimum improvement threshold
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Stratification (for continuous target)
    "n_quantiles": 5,          # Stratify by quintiles
    
    # Random seed
    "random_seed": RANDOM_SEED
}


def adjust_config_for_small_dataset(config, n_samples):
    """
    Adjust hyperparameters for small datasets to prevent overfitting.
    For N < 1000: increase regularization, reduce model complexity.
    """
    if n_samples < 1000:
        print(f"\n⚠ Small dataset detected (N={n_samples})")
        print("Adjusting hyperparameters to prevent overfitting:")
        
        # Increase dropout significantly
        config["dropout_genomic"] = 0.6
        config["dropout_drug"] = 0.5
        config["dropout_head"] = 0.6
        print(f"  • Dropout: genomic={config['dropout_genomic']}, drug={config['dropout_drug']}, head={config['dropout_head']}")
        
        # Increase weight decay (L2 regularization)
        config["weight_decay"] = 5e-3
        print(f"  • Weight decay: {config['weight_decay']}")
        
        # Reduce learning rate slightly
        config["learning_rate"] = 5e-4
        print(f"  • Learning rate: {config['learning_rate']}")
        
        # More aggressive early stopping
        config["patience"] = 10
        print(f"  • Early stopping patience: {config['patience']}")
        
        # Smaller batch size for better generalization
        config["batch_size"] = 16
        print(f"  • Batch size: {config['batch_size']}")
        
        print()
    
    return config


# ===================================================================
# DATASET CLASS
# ===================================================================

class DrugResponseDataset(Dataset):
    """PyTorch dataset for drug response prediction."""
    
    def __init__(self, X_genomic, X_drug, y):
        self.X_genomic = torch.FloatTensor(X_genomic)
        self.X_drug = torch.FloatTensor(X_drug)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X_genomic[idx], self.X_drug[idx], self.y[idx]


# ===================================================================
# DATA LOADING AND STRATIFIED SPLITTING
# ===================================================================

def create_stratified_bins(y, n_quantiles=5):
    """
    Create stratification bins for continuous target variable.
    
    Evidence: Riley et al. 2020 (BMJ) - Stratify continuous outcomes by quantiles
    for regression tasks to ensure balanced distribution across splits.
    
    Args:
        y: Target values (continuous)
        n_quantiles: Number of quantiles for stratification
        
    Returns:
        bins: Stratification labels for each sample
    """
    quantiles = np.linspace(0, 100, n_quantiles + 1)
    thresholds = np.percentile(y, quantiles)
    bins = np.digitize(y, thresholds[1:-1])  # Assign to bins
    
    print(f"\nStratification bins ({n_quantiles} quantiles):")
    for i in range(n_quantiles):
        mask = bins == i
        print(f"  Bin {i}: {mask.sum():4d} samples, y ∈ [{y[mask].min():.3f}, {y[mask].max():.3f}]")
    
    return bins


def load_and_split_data(config):
    """
    Load preprocessed data and create PATIENT-LEVEL stratified train/val/test splits.
    
    CRITICAL: Prevents data leakage by ensuring all treatments from the same patient
    stay together in the same split (train/val/test).
    
    Evidence-based split strategy:
    - Collins et al. 2015 (J Clin Epidemiol): 70/15/15 recommended for N=3000-5000
    - Riley et al. 2020 (BMJ): Stratified sampling for continuous outcomes
    - Bouthillier et al. 2021 (NeurIPS): Fixed random seed for reproducibility
    - Wiens et al. 2014 (AMIA): Patient-level splitting prevents overfitting
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
        datasets: Dict with split sizes and distributions
    """
    print("=" * 70)
    print("LOADING AND SPLITTING DATA (PATIENT-LEVEL)")
    print("=" * 70)
    
    # Load preprocessed features (CLEANED - no patient identifiers)
    X_tabular = np.load('artifacts/X_tabular_clean.npy')
    X_drug_fp = np.load('artifacts/X_drug_fp_clean.npy')
    y = np.load('artifacts/y_clean.npy')  # Fixed: was y_pfs_clean.npy
    patient_ids = np.load('artifacts/patient_ids_clean.npy', allow_pickle=True)
    
    # Filter for PREVIOUS TREATMENT ONLY (treatment_line >= 2)
    print("\n🔍 FILTERING FOR PREVIOUS TREATMENT ONLY (treatment_line >= 2)")
    print("   First treatment uses XGBoost (see train_xgboost.py)")
    treatment_lines = X_tabular[:, 0]
    mask = treatment_lines >= 2.0
    
    X_tabular = X_tabular[mask]
    X_drug_fp = X_drug_fp[mask]
    y = y[mask]
    patient_ids = patient_ids[mask]
    
    print(f"  ✅ Kept: {mask.sum()} samples from treatment line 2+")
    print(f"  ❌ Filtered out: {(~mask).sum()} samples from first treatment")
    
    print(f"\nLoaded data:")
    print(f"  X_tabular: {X_tabular.shape} (genomic + clinical + microbiome)")
    print(f"  X_drug_fp: {X_drug_fp.shape} (Morgan fingerprints)")
    print(f"  y: {y.shape} (effectiveness scores)")
    print(f"  Target range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Target mean: {y.mean():.3f} ± {y.std():.3f}")
    
    # Adjust config for small datasets
    config = adjust_config_for_small_dataset(config, len(y))
    
    # Check for patient-level structure
    unique_patients = np.unique(patient_ids)
    print(f"\nPatient-level statistics:")
    print(f"  Total samples: {len(y)}")
    print(f"  Unique patients: {len(unique_patients)}")
    print(f"  Avg treatments per patient: {len(y) / len(unique_patients):.2f}")
    
    # Create patient-level target for stratification
    # Use median PFS per patient for stratification to balance outcomes
    import pandas as pd
    patient_df = pd.DataFrame({
        'patient_id': patient_ids,
        'y': y
    })
    patient_medians = patient_df.groupby('patient_id')['y'].median().values
    
    print(f"  Patient median PFS range: [{patient_medians.min():.3f}, {patient_medians.max():.3f}]")
    
    # Create stratification bins at PATIENT level
    quantiles = np.linspace(0, 100, config["n_quantiles"] + 1)
    thresholds = np.percentile(patient_medians, quantiles)
    patient_bins = np.digitize(patient_medians, thresholds[1:-1])
    
    print(f"\nPatient-level stratification bins ({config['n_quantiles']} quantiles):")
    for i in range(config["n_quantiles"]):
        mask = patient_bins == i
        n_patients = mask.sum()
        n_samples = (patient_df.groupby('patient_id')['y'].count()[patient_bins == i]).sum()
        print(f"  Bin {i}: {n_patients:4d} patients ({n_samples:4d} samples), "
              f"median PFS in [{patient_medians[mask].min():.3f}, {patient_medians[mask].max():.3f}]")
    
    # ===================================================================
    # PATIENT-LEVEL SPLITTING: All treatments from same patient stay together
    # ===================================================================
    
    # Split 1: Separate test patients (15%)
    # Evidence: Wiens et al. 2014 - Patient-level splitting prevents leakage
    patients_trainval, patients_test = train_test_split(
        unique_patients,
        test_size=config["test_size"],
        random_state=config["random_seed"],
        stratify=patient_bins
    )
    
    # Split 2: Separate validation patients from training (15% of remaining)
    patient_bins_trainval = patient_bins[np.isin(unique_patients, patients_trainval)]
    val_ratio = config["val_size"] / (config["train_size"] + config["val_size"])
    patients_train, patients_val = train_test_split(
        patients_trainval,
        test_size=val_ratio,
        random_state=config["random_seed"],
        stratify=patient_bins_trainval
    )
    
    print(f"\nPatient-level splits:")
    print(f"  Training patients: {len(patients_train)}")
    print(f"  Validation patients: {len(patients_val)}")
    print(f"  Test patients: {len(patients_test)}")
    
    # Map patients to sample indices
    train_mask = np.isin(patient_ids, patients_train)
    val_mask = np.isin(patient_ids, patients_val)
    test_mask = np.isin(patient_ids, patients_test)
    
    # Extract samples for each split
    X_tab_trainval = X_tabular[train_mask | val_mask]
    X_drug_trainval = X_drug_fp[train_mask | val_mask]
    y_trainval = y[train_mask | val_mask]
    
    X_tab_test = X_tabular[test_mask]
    X_drug_test = X_drug_fp[test_mask]
    y_test = y[test_mask]
    
    X_tab_train = X_tabular[train_mask]
    X_drug_train = X_drug_fp[train_mask]
    y_train = y[train_mask]
    
    X_tab_val = X_tabular[val_mask]
    X_drug_val = X_drug_fp[val_mask]
    y_val = y[val_mask]
    
    # Verify no patient overlap
    print(f"\nVerifying patient-level separation:")
    train_patients_set = set(patient_ids[train_mask])
    val_patients_set = set(patient_ids[val_mask])
    test_patients_set = set(patient_ids[test_mask])
    
    overlap_train_val = train_patients_set & val_patients_set
    overlap_train_test = train_patients_set & test_patients_set
    overlap_val_test = val_patients_set & test_patients_set
    
    if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
        print(f"  * No patient overlap between splits!")
        print(f"  * Train patients: {len(train_patients_set)}")
        print(f"  * Val patients: {len(val_patients_set)}")
        print(f"  * Test patients: {len(test_patients_set)}")
    else:
        print(f"  WARNING: Patient overlap detected!")
        print(f"    Train-Val overlap: {len(overlap_train_val)} patients")
        print(f"    Train-Test overlap: {len(overlap_train_test)} patients")
        print(f"    Val-Test overlap: {len(overlap_val_test)} patients")
    
    # ===================================================================
    # DATA CLEANING: Handle NaN, Inf, and extreme values
    # ===================================================================
    print("\n" + "=" * 70)
    print("DATA CLEANING")
    print("=" * 70)
    
    # Check for NaN/Inf BEFORE cleaning
    print("\nBefore cleaning:")
    for name, arr in [("X_tab_train", X_tab_train), ("X_tab_val", X_tab_val), 
                      ("X_tab_test", X_tab_test), ("X_drug_train", X_drug_train),
                      ("X_drug_val", X_drug_val), ("X_drug_test", X_drug_test),
                      ("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"  {name}: {n_nan} NaNs, {n_inf} Infs")
    
    # Replace NaN/Inf with feature-wise medians from training set (better imputation)
    print("\nImputing missing values with feature-wise medians...")
    
    # Compute medians from training set (ignoring NaNs)
    X_tab_medians = np.nanmedian(X_tab_train, axis=0)
    X_drug_medians = np.nanmedian(X_drug_train, axis=0)
    
    # Replace any remaining NaNs in medians with 0 (for features that are all NaN)
    X_tab_medians = np.nan_to_num(X_tab_medians, nan=0.0)
    X_drug_medians = np.nan_to_num(X_drug_medians, nan=0.0)
    
    # Apply imputation to each split
    for i in range(X_tab_train.shape[1]):
        X_tab_train[np.isnan(X_tab_train[:, i]), i] = X_tab_medians[i]
        X_tab_val[np.isnan(X_tab_val[:, i]), i] = X_tab_medians[i]
        X_tab_test[np.isnan(X_tab_test[:, i]), i] = X_tab_medians[i]
    
    for i in range(X_drug_train.shape[1]):
        X_drug_train[np.isnan(X_drug_train[:, i]), i] = X_drug_medians[i]
        X_drug_val[np.isnan(X_drug_val[:, i]), i] = X_drug_medians[i]
        X_drug_test[np.isnan(X_drug_test[:, i]), i] = X_drug_medians[i]
    
    # Replace any remaining Inf values with 0
    X_tab_train = np.nan_to_num(X_tab_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_tab_val = np.nan_to_num(X_tab_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_tab_test = np.nan_to_num(X_tab_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_drug_train = np.nan_to_num(X_drug_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_drug_val = np.nan_to_num(X_drug_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_drug_test = np.nan_to_num(X_drug_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Replace NaN/Inf in targets with median (should not happen but safeguard)
    y_median = np.median(y_train)
    y_train = np.nan_to_num(y_train, nan=y_median, posinf=y_median, neginf=y_median)
    y_val = np.nan_to_num(y_val, nan=y_median, posinf=y_median, neginf=y_median)
    y_test = np.nan_to_num(y_test, nan=y_median, posinf=y_median, neginf=y_median)
    
    # Clip extreme values to prevent numerical instability
    # Based on training set statistics
    X_tab_max = np.abs(X_tab_train).max()
    X_drug_max = np.abs(X_drug_train).max()
    clip_threshold = max(100.0, X_tab_max * 3)  # 3x max as safety
    
    X_tab_train = np.clip(X_tab_train, -clip_threshold, clip_threshold)
    X_tab_val = np.clip(X_tab_val, -clip_threshold, clip_threshold)
    X_tab_test = np.clip(X_tab_test, -clip_threshold, clip_threshold)
    
    X_drug_train = np.clip(X_drug_train, -clip_threshold, clip_threshold)
    X_drug_val = np.clip(X_drug_val, -clip_threshold, clip_threshold)
    X_drug_test = np.clip(X_drug_test, -clip_threshold, clip_threshold)
    
    # Verify cleaning worked
    print("\nAfter cleaning:")
    all_clean = True
    for name, arr in [("X_tab_train", X_tab_train), ("X_tab_val", X_tab_val), 
                      ("X_tab_test", X_tab_test), ("X_drug_train", X_drug_train),
                      ("X_drug_val", X_drug_val), ("X_drug_test", X_drug_test),
                      ("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"{name}: STILL HAS {n_nan} NaNs, {n_inf} Infs")
            all_clean = False
    
    if all_clean:
        print("  * All arrays clean (no NaN/Inf)")
    
    print(f"\nData ranges after cleaning:")
    print(f"  X_tabular: [{X_tab_train.min():.2f}, {X_tab_train.max():.2f}]")
    print(f"  X_drug_fp: [{X_drug_train.min():.2f}, {X_drug_train.max():.2f}]")
    print(f"  y_train: [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    # ===================================================================
    # Print split statistics
    # ===================================================================
    print(f"\n{'Split':<15} {'N Patients':>12} {'N Samples':>12} {'%':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 90)
    
    splits_info = [
        ("Training", len(train_patients_set), len(y_train), y_train),
        ("Validation", len(val_patients_set), len(y_val), y_val),
        ("Test", len(test_patients_set), len(y_test), y_test)
    ]
    
    for name, n_patients, n_samples, y_split in splits_info:
        pct = 100 * n_samples / len(y)
        print(f"{name:<15} {n_patients:>12} {n_samples:>12} {pct:>5.1f}% {y_split.mean():>8.3f} "
              f"{y_split.std():>8.3f} {y_split.min():>8.3f} {y_split.max():>8.3f}")
    
    print("\n" + "=" * 90)
    print("Patient-level split: Target 6-month AUROC >= 0.65 (acceptable), >= 0.70 (good)")
    print("=" * 90)
    
    # Create PyTorch datasets
    train_dataset = DrugResponseDataset(X_tab_train, X_drug_train, y_train)
    val_dataset = DrugResponseDataset(X_tab_val, X_drug_val, y_val)
    test_dataset = DrugResponseDataset(X_tab_test, X_drug_test, y_test)
    
    # Create data loaders
    # Evidence: Way et al. 2018 - Batch size 32-64, shuffle training only
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,  # Shuffle training data
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,  # Don't shuffle validation
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,  # Don't shuffle test
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Save split info
    split_info = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "train_pct": 100 * len(train_dataset) / len(y),
        "val_pct": 100 * len(val_dataset) / len(y),
        "test_pct": 100 * len(test_dataset) / len(y),
        "random_seed": config["random_seed"],
        "stratification": f"{config['n_quantiles']} quantiles"
    }
    
    return train_loader, val_loader, test_loader, split_info


# ===================================================================
# TRAINING LOOP
# ===================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for X_genomic, X_drug, y_true in loader:
        X_genomic = X_genomic.to(device)
        X_drug = X_drug.to(device)
        y_true = y_true.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        y_pred = model(X_genomic, X_drug)
        loss = criterion(y_pred, y_true)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (Pascanu et al. 2013)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_genomic, X_drug, y_true in loader:
            X_genomic = X_genomic.to(device)
            X_drug = X_drug.to(device)
            y_true = y_true.to(device)
            
            y_pred = model(X_genomic, X_drug)
            loss = criterion(y_pred, y_true)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # R² score
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Pearson correlation
    corr = np.corrcoef(all_preds, all_targets)[0, 1]
    
    metrics = {
        "loss": total_loss / n_batches,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "pearson_r": corr
    }
    
    return metrics, all_preds, all_targets


def train_model(model, train_loader, val_loader, config):
    """
    Train model with early stopping and checkpointing.
    
    Evidence:
    - Prechelt 1998 (Neural Networks): Early stopping prevents overfitting
    - Loshchilov & Hutter 2017 (ICLR): Cosine annealing learning rate
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    device = torch.device(config["device"])
    model = model.to(device)
    
    # Loss function (Huber loss for robustness to outliers)
    criterion = nn.HuberLoss(delta=1.0)
    
    # Optimizer and scheduler
    optimizer = create_optimizer(
        model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = create_scheduler(
        optimizer,
        num_epochs=config["num_epochs"],
        warmup_epochs=config["warmup_epochs"]
    )
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_r2": [],
        "val_pearson": [],
        "learning_rate": []
    }
    
    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    
    print(f"\nDevice: {device}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"Early stopping patience: {config['patience']} epochs")
    print(f"Warmup epochs: {config['warmup_epochs']}")
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(config["num_epochs"]):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_r2"].append(val_metrics["r2"])
        history["val_pearson"].append(val_metrics["pearson_r"])
        history["learning_rate"].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{config['num_epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val MAE: {val_metrics['mae']:.4f} | "
              f"Val R²: {val_metrics['r2']:.4f} | "
              f"Val r: {val_metrics['pearson_r']:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Early stopping check
        if val_metrics["loss"] < (best_val_loss - config["min_delta"]):
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "config": config
            }, CHECKPOINT_DIR / "best_model.pt")
            print(f"  Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best model was at epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")
                break
    
    # Save final model
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "config": config
    }, CHECKPOINT_DIR / "final_model.pt")
    
    # Save training history (convert numpy types to Python native)
    history_serializable = {
        key: [float(v) if hasattr(v, 'item') else v for v in values]
        for key, values in history.items()
    }
    with open(CHECKPOINT_DIR / "training_history.json", "w") as f:
        json.dump(history_serializable, f, indent=2)
    
    return history, best_epoch


# ===================================================================
# EVALUATION AND VISUALIZATION
# ===================================================================

def plot_training_history(history, save_path):
    """
    Plot training curves with overfitting diagnostics.
    
    Key metrics to watch:
    - Train/Val Loss gap: Large gap = overfitting
    - Val loss increasing while train decreasing = overfitting
    - Val R² plateau or decrease = overfitting
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # 1. Loss curves with gap highlighting
    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2, color='blue')
    axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2, color='red')
    axes[0, 0].fill_between(epochs, history["train_loss"], history["val_loss"], 
                            alpha=0.3, color='yellow', label='Train-Val Gap')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss (Huber)")
    axes[0, 0].set_title("Loss: Train vs Validation\n(Large gap = Overfitting)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Loss ratio (val/train) - should stay near 1.0
    loss_ratio = [v/t if t > 0 else 1.0 for v, t in zip(history["val_loss"], history["train_loss"])]
    axes[0, 1].plot(epochs, loss_ratio, color="purple", linewidth=2)
    axes[0, 1].axhline(y=1.0, color='black', linestyle='--', label='Perfect ratio')
    axes[0, 1].axhline(y=1.2, color='orange', linestyle=':', label='Mild overfitting')
    axes[0, 1].axhline(y=1.5, color='red', linestyle=':', label='Severe overfitting')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Val Loss / Train Loss")
    axes[0, 1].set_title("Overfitting Ratio\n(>1.2 = Overfitting)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim([0.8, 2.0])
    
    # 3. MAE
    axes[0, 2].plot(epochs, history["val_mae"], color="green", linewidth=2)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("MAE (log-space)")
    axes[0, 2].set_title("Validation MAE")
    axes[0, 2].grid(alpha=0.3)
    
    # 4. R² score with realistic benchmarks
    axes[1, 0].plot(epochs, history["val_r2"], color="purple", linewidth=2)
    axes[1, 0].axhline(y=0.50, color='green', linestyle='--', alpha=0.5, label='Good (0.50)')
    axes[1, 0].axhline(y=0.70, color='blue', linestyle='--', alpha=0.5, label='Excellent (0.70)')
    axes[1, 0].axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='Suspicious (0.90)')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("R²")
    axes[1, 0].set_title("Validation R²\n(>0.90 may indicate leakage)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_ylim([0, 1.0])
    
    # 5. Pearson correlation
    axes[1, 1].plot(epochs, history["val_pearson"], color="orange", linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Pearson r")
    axes[1, 1].set_title("Validation Pearson Correlation")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_ylim([0, 1.0])
    
    # 6. Learning rate
    axes[1, 2].plot(epochs, history["learning_rate"], color="brown", linewidth=2)
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Learning Rate")
    axes[1, 2].set_title("Learning Rate Schedule")
    axes[1, 2].set_yscale("log")
    axes[1, 2].grid(alpha=0.3)
    
    # Add overall title with diagnostics
    best_val_r2 = max(history["val_r2"])
    final_val_r2 = history["val_r2"][-1]
    final_loss_ratio = loss_ratio[-1]
    
    diagnosis = ""
    if final_loss_ratio > 1.5:
        diagnosis = "SEVERE OVERFITTING"
    elif final_loss_ratio > 1.2:
        diagnosis = "MILD OVERFITTING"
    elif best_val_r2 > 0.90:
        diagnosis = "SUSPICIOUSLY HIGH R² (check for leakage)"
    elif final_val_r2 < best_val_r2 - 0.05:
        diagnosis = "OVERFITTING (val R² declining)"
    else:
        diagnosis = "HEALTHY TRAINING"
    
    fig.suptitle(f'Training Diagnostics | Final Val R²: {final_val_r2:.3f} | Status: {diagnosis}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nTraining diagnostics saved to {save_path}")
    print(f"  Final Val Loss / Train Loss ratio: {final_loss_ratio:.3f}")
    print(f"  Diagnosis: {diagnosis}")


def test_model(model, test_loader, config):
    """Final evaluation on held-out test set."""
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    device = torch.device(config["device"])
    
    # Load best model (weights_only=False for PyTorch 2.6 compatibility)
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    criterion = nn.HuberLoss(delta=1.0)
    metrics, preds, targets = evaluate(model, test_loader, criterion, device)
    
    # Inverse transform: PFS_months = exp(prediction) - 1
    preds_months = np.exp(preds) - 1
    targets_months = np.exp(targets) - 1
    
    # Calculate basic regression metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import pearsonr
    
    mae_months = mean_absolute_error(targets_months, preds_months)
    r2_months = r2_score(targets_months, preds_months)
    r_months, _ = pearsonr(targets_months, preds_months)
    
    print(f"\nRegression Metrics (months): MAE={mae_months:.2f}, R²={r2_months:.4f}")
    
    # Binary classification metrics - PRIMARY EVALUATION METRIC
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, roc_curve, confusion_matrix
    
    print("\n" + "="*70)
    print("CLASSIFICATION METRICS")
    print("="*70)
    
    # Define thresholds - 6 months is primary
    thresholds_months = {
        "6 months (PRIMARY - clinical benefit)": 6.0,
        "3 months (early progression)": 3.0,
        "Median (2.53 months)": np.median(targets_months),
    }
    
    classification_results = {}
    
    for i, (threshold_name, threshold) in enumerate(thresholds_months.items()):
        # Convert to binary labels
        y_true_binary = (targets_months >= threshold).astype(int)
        y_pred_binary = (preds_months >= threshold).astype(int)
        
        # Calculate metrics
        auroc = roc_auc_score(y_true_binary, preds_months)
        auprc = average_precision_score(y_true_binary, preds_months)
        f1 = f1_score(y_true_binary, y_pred_binary)
        balanced_acc = balanced_accuracy_score(y_true_binary, y_pred_binary)
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        # Find optimal threshold using Youden's J statistic
        fpr, tpr, thresholds_roc = roc_curve(y_true_binary, preds_months)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold_pred = thresholds_roc[optimal_idx]
        optimal_sensitivity = tpr[optimal_idx]
        optimal_specificity = 1 - fpr[optimal_idx]
        
        # Predict using optimal threshold
        y_pred_optimal = (preds_months >= optimal_threshold_pred).astype(int)
        cm_optimal = confusion_matrix(y_true_binary, y_pred_optimal)
        tn_opt, fp_opt, fn_opt, tp_opt = cm_optimal.ravel()
        f1_optimal = f1_score(y_true_binary, y_pred_optimal)
        balanced_acc_optimal = balanced_accuracy_score(y_true_binary, y_pred_optimal)
        
        # Class distribution
        n_positive = y_true_binary.sum()
        n_negative = len(y_true_binary) - n_positive
        
        # Emphasize 6-month threshold
        if i == 0:  # Primary threshold
            print(f"\n{'*'*70}")
            print(f"\n*** PRIMARY: {threshold_name} ***")
        else:
            print(f"\n{threshold_name}:")
        
        print(f"  AUROC: {auroc:.4f}", end="")
        if auroc >= 0.75:
            print(" (STRONG)", end="")
        elif auroc >= 0.65:
            print(" (Acceptable)", end="")
        else:
            print(" (Below target)", end="")
        print(f", N+={n_positive}, N-={n_negative}")
        
        # Unpack confusion matrices for storage
        tn, fp, fn, tp = cm.ravel()
        
        classification_results[threshold_name] = {
            'threshold_months': threshold,
            'auroc': auroc,
            'auprc': auprc,
            'f1': f1,
            'f1_optimal': f1_optimal,
            'balanced_acc': balanced_acc,
            'balanced_acc_optimal': balanced_acc_optimal,
            'optimal_pred_threshold': float(optimal_threshold_pred),
            'optimal_sensitivity': float(optimal_sensitivity),
            'optimal_specificity': float(optimal_specificity),
            'n_positive': int(n_positive),
            'n_negative': int(n_negative),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'confusion_matrix_optimal': {'tn': int(tn_opt), 'fp': int(fp_opt), 'fn': int(fn_opt), 'tp': int(tp_opt)}
        }
    
    # Highlight primary result
    primary_auroc = classification_results["6 months (PRIMARY - clinical benefit)"]['auroc']
    primary_optimal = classification_results["6 months (PRIMARY - clinical benefit)"]
    
    print(f"\n{'='*70}")
    print(f"PRIMARY METRIC: 6-Month AUROC = {primary_auroc:.4f}")
    if primary_auroc >= 0.75:
        print("Performance: STRONG")
    elif primary_auroc >= 0.65:
        print("Performance: ACCEPTABLE")
    else:
        print("Performance: NEEDS IMPROVEMENT (target >=0.65)")
    print(f"{'='*70}")
    
    # Save predictions (both spaces)
    np.save(CHECKPOINT_DIR / "test_predictions_log.npy", preds)
    np.save(CHECKPOINT_DIR / "test_targets_log.npy", targets)
    np.save(CHECKPOINT_DIR / "test_predictions_months.npy", preds_months)
    np.save(CHECKPOINT_DIR / "test_targets_months.npy", targets_months)
    
    # Save classification results (convert numpy types to Python native)
    import json
    classification_results_serializable = {}
    for key, val in classification_results.items():
        classification_results_serializable[key] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
            for k, v in val.items()
        }
    
    with open(CHECKPOINT_DIR / "classification_metrics.json", "w") as f:
        json.dump(classification_results_serializable, f, indent=2)
    
    # Plot ROC curves for all thresholds
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ROC Curves
    ax1 = axes[0]
    for threshold_name, threshold in thresholds_months.items():
        y_true_binary = (targets_months >= threshold).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, preds_months)
        auroc = classification_results[threshold_name]['auroc']
        ax1.plot(fpr, tpr, linewidth=2, label=f'{threshold_name} (AUROC={auroc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    ax1.axhline(y=0.65, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Acceptable (0.65)')
    ax1.axhline(y=0.75, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Strong (0.75)')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves: Drug Effectiveness Classification')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Regression scatter plot
    ax2 = axes[1]
    ax2.scatter(targets_months, preds_months, alpha=0.5, s=20, c='steelblue')
    ax2.plot([targets_months.min(), targets_months.max()], 
             [targets_months.min(), targets_months.max()],
             'r--', linewidth=2, label="Perfect prediction")
    
    # Add threshold lines
    for threshold_name, threshold in thresholds_months.items():
        ax2.axhline(y=threshold, color='gray', linestyle='--', linewidth=1, alpha=0.3)
        ax2.axvline(x=threshold, color='gray', linestyle='--', linewidth=1, alpha=0.3)
        ax2.text(0, threshold+0.5, f'{threshold:.1f}mo', fontsize=8, alpha=0.6)
    
    ax2.set_xlabel("Actual PFS (months)")
    ax2.set_ylabel("Predicted PFS (months)")
    ax2.set_title(f"Regression: PFS = exp(prediction) - 1\nR² = {r2_months:.3f}")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / "test_predictions_with_roc.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nROC curves and predictions saved to {CHECKPOINT_DIR / 'test_predictions_with_roc.png'}")

    # Plot predictions vs actual (both spaces)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Log space
    ax1 = axes[0]
    ax1.scatter(targets, preds, alpha=0.5, s=20)
    ax1.plot([targets.min(), targets.max()], [targets.min(), targets.max()],
             'r--', linewidth=2, label="Perfect prediction")
    ax1.set_xlabel("Actual log(PFS+1)")
    ax1.set_ylabel("Predicted log(PFS+1)")
    ax1.set_title(f"Log Space\nR² = {metrics['r2']:.3f}, r = {metrics['pearson_r']:.3f}")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Original space (months)
    ax2 = axes[1]
    ax2.scatter(targets_months, preds_months, alpha=0.5, s=20)
    ax2.plot([targets_months.min(), targets_months.max()], 
             [targets_months.min(), targets_months.max()],
             'r--', linewidth=2, label="Perfect prediction")
    ax2.set_xlabel("Actual PFS (months)")
    ax2.set_ylabel("Predicted PFS (months)")
    ax2.set_title(f"Original Space: PFS = exp(prediction) - 1\nR² = {r2_months:.3f}, r = {r_months:.3f}")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / "test_predictions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPrediction plots saved to {CHECKPOINT_DIR / 'test_predictions.png'}")
    
    # Store both metrics
    metrics['r2_months'] = r2_months
    metrics['mae_months'] = mae_months
    metrics['pearson_r_months'] = r_months
    metrics['classification'] = classification_results
    
    return metrics


def analyze_overfitting(history):
    """
    Analyze training history to identify when overfitting begins.
    """
    print("\n" + "=" * 70)
    print("OVERFITTING ANALYSIS")
    print("=" * 70)
    
    train_loss = np.array(history["train_loss"])
    val_loss = np.array(history["val_loss"])
    val_r2 = np.array(history["val_r2"])
    
    # Calculate loss ratio
    loss_ratio = val_loss / train_loss
    
    # Find when overfitting starts (multiple criteria)
    overfitting_signals = []
    
    # Criterion 1: Val loss starts increasing while train loss decreases
    for i in range(5, len(val_loss)):
        if val_loss[i] > val_loss[i-1] and train_loss[i] < train_loss[i-1]:
            overfitting_signals.append(("Val loss increasing", i+1))
            break
    
    # Criterion 2: Loss ratio exceeds 1.2
    for i, ratio in enumerate(loss_ratio):
        if ratio > 1.2:
            overfitting_signals.append(("Loss ratio > 1.2", i+1))
            break
    
    # Criterion 3: Val R² starts declining from peak
    best_r2_idx = np.argmax(val_r2)
    best_r2 = val_r2[best_r2_idx]
    for i in range(best_r2_idx + 5, len(val_r2)):
        if val_r2[i] < best_r2 - 0.05:
            overfitting_signals.append(("Val R² declining", i+1))
            break
    
    print("\nOverfitting Detection:")
    if overfitting_signals:
        print(f"  * Overfitting detected at epoch {min(s[1] for s in overfitting_signals)}")
        for signal, epoch in overfitting_signals:
            print(f"    - {signal} at epoch {epoch}")
    else:
        print("  * No clear overfitting detected in training history")
    
    # Best epoch analysis
    print(f"\nBest Performance:")
    print(f"  Epoch {best_r2_idx + 1}: Val R² = {best_r2:.4f}, Val Loss = {val_loss[best_r2_idx]:.4f}")
    print(f"  Loss ratio at best epoch: {loss_ratio[best_r2_idx]:.3f}")
    
    # Final epoch analysis
    print(f"\nFinal Epoch:")
    print(f"  Epoch {len(val_r2)}: Val R² = {val_r2[-1]:.4f}, Val Loss = {val_loss[-1]:.4f}")
    print(f"  Loss ratio at final epoch: {loss_ratio[-1]:.3f}")
    
    # Overall assessment
    final_ratio = loss_ratio[-1]
    final_r2 = val_r2[-1]
    
    print("\n" + "=" * 70)
    print("VERDICT:")
    print("=" * 70)
    
    if final_r2 > 0.95:
        print("  ⚠️  CRITICAL: R² = {:.3f} is EXTREMELY HIGH!".format(final_r2))
        print("      This strongly suggests DATA LEAKAGE.")
        print("      For new patients, R² should be 0.3-0.7 (published benchmarks)")
        print()
        print("  POSSIBLE CAUSES:")
        print("      1. Same patients in train and validation (MOST LIKELY)")
        print("      2. Target variable leaked into features")
        print("      3. Temporal leakage (future info in features)")
    elif final_r2 > 0.85:
        print("  ⚠️  WARNING: R² = {:.3f} is very high".format(final_r2))
        print("      Possible mild data leakage or very easy prediction task")
    elif final_ratio > 1.5:
        print("  ⚠️  SEVERE OVERFITTING: Val/Train loss ratio = {:.3f}".format(final_ratio))
        print("      Model is memorizing training data")
        print()
        print("  RECOMMENDATIONS:")
        print("      1. Increase weight_decay (current: 1e-4 → try 1e-3)")
        print("      2. Increase dropout (current: 0.3-0.4 → try 0.5-0.6)")
        print("      3. Reduce model capacity (fewer layers or smaller dimensions)")
        print("      4. Use more aggressive data augmentation")
    elif final_ratio > 1.2:
        print("  ⚠️  MILD OVERFITTING: Val/Train loss ratio = {:.3f}".format(final_ratio))
        print("      Some memorization occurring")
        print()
        print("  RECOMMENDATIONS:")
        print("      1. Slightly increase regularization")
        print("      2. Early stopping is working correctly")
    else:
        print("  ✓ HEALTHY TRAINING: Val/Train loss ratio = {:.3f}".format(final_ratio))
        print("      Model is generalizing well")
        print("      R² = {:.3f} is reasonable for survival prediction".format(final_r2))
    
    print("=" * 70)


# ===================================================================
# MAIN EXECUTION
# ===================================================================

def main():
    """Main training pipeline for PREVIOUS TREATMENT data."""
    
    print("\n" + "=" * 70)
    print("TRAINING: PREVIOUS TREATMENT (line >= 2) - Deep Learning")
    print("=" * 70)
    print("Note: First treatment uses XGBoost (train_xgboost.py)")
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"previous_treatment_{timestamp}.log"
    
    # Load and split data
    train_loader, val_loader, test_loader, split_info = load_and_split_data(CONFIG)
    
    # Save configuration
    with open(CHECKPOINT_DIR / "config.json", "w") as f:
        json.dump({**CONFIG, **split_info}, f, indent=2)
    
    # Initialize model
    model = ImprovedDrugResponseModel(
        genomic_dim=CONFIG["genomic_dim"],
        drug_fp_dim=CONFIG["drug_fp_dim"],
        embed_dim=CONFIG["embed_dim"],
        dropout_genomic=CONFIG["dropout_genomic"],
        dropout_drug=CONFIG["dropout_drug"],
        dropout_head=CONFIG["dropout_head"]
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized with {total_params:,} parameters")
    
    # Train model
    history, best_epoch = train_model(model, train_loader, val_loader, CONFIG)
    
    # Analyze overfitting
    analyze_overfitting(history)
    
    # Plot training curves
    plot_training_history(history, CHECKPOINT_DIR / "training_curves.png")
    
    # Final evaluation on test set
    test_metrics = test_model(model, test_loader, CONFIG)
    
    # Save training log
    log_data = {
        'timestamp': timestamp,
        'model_type': 'deep_learning',
        'stratum': 'previous_treatment',
        'n_samples': split_info.get('n_samples', 0),
        'split_info': split_info,
        'best_epoch': best_epoch,
        'final_metrics': test_metrics,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n✅ Training log saved to: {log_file}")
    print(f"✅ Model checkpoint saved to: {CHECKPOINT_DIR / 'best_model.pt'}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best model saved at epoch {best_epoch}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print(f"Final test R² = {test_metrics['r2']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
