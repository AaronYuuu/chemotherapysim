"""
Training script for DeepSurv model with multimodal cancer data.

Uses Cox proportional hazards loss for proper survival analysis.
Evaluates using C-index (concordance index), which is equivalent to AUROC for survival models.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error
from scipy.stats import spearmanr
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.deepsurv_model import DeepSurv, CoxPHLoss, ConcordanceIndex, create_event_indicator


class MultimodalDataset(Dataset):
    """Dataset for genomic + drug + clinical data"""
    def __init__(self, genomic, drug, clinical, targets, events=None):
        self.genomic = torch.FloatTensor(genomic)
        self.drug = torch.FloatTensor(drug)
        self.clinical = torch.FloatTensor(clinical)
        self.targets = torch.FloatTensor(targets)
        
        # Use provided event indicators if available (from pfs_status)
        # Otherwise create binary indicators based on median PFS
        if events is not None:
            self.events = torch.FloatTensor(events)
            self.median_pfs = np.median(targets)
        else:
            events, self.median_pfs = create_event_indicator(targets)
            self.events = torch.FloatTensor(events)
    
    def __len__(self):
        return len(self.genomic)
    
    def __getitem__(self, idx):
        return {
            'genomic': self.genomic[idx],
            'drug': self.drug[idx],
            'clinical': self.clinical[idx],
            'target': self.targets[idx],
            'event': self.events[idx]
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        genomic = batch['genomic'].to(device)
        drug = batch['drug'].to(device)
        clinical = batch['clinical'].to(device)
        times = batch['target'].to(device)
        events = batch['event'].to(device)
        
        # Forward pass
        risk_scores = model(genomic, drug, clinical)
        
        # Cox loss
        loss = criterion(risk_scores, times, events)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, dataset_median_pfs=None):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_risk_scores = []
    all_targets = []
    all_events = []
    
    with torch.no_grad():
        for batch in dataloader:
            genomic = batch['genomic'].to(device)
            drug = batch['drug'].to(device)
            clinical = batch['clinical'].to(device)
            times = batch['target'].to(device)
            events = batch['event'].to(device)
            
            # Forward pass
            risk_scores = model(genomic, drug, clinical)
            
            # Loss
            loss = criterion(risk_scores, times, events)
            total_loss += loss.item()
            
            # Collect predictions
            all_risk_scores.append(risk_scores.cpu())
            all_targets.append(times.cpu())
            all_events.append(events.cpu())
    
    # Concatenate all predictions
    all_risk_scores = torch.cat(all_risk_scores).squeeze()
    all_targets = torch.cat(all_targets)
    all_events = torch.cat(all_events)
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    # Convert risk scores to PFS predictions (inverse relationship)
    # Higher risk = shorter PFS
    pfs_predictions = -all_risk_scores  # Invert for comparison
    
    # MAE (on PFS predictions)
    # Scale predictions to match target range
    pred_min, pred_max = pfs_predictions.min(), pfs_predictions.max()
    target_min, target_max = all_targets.min(), all_targets.max()
    
    if pred_max != pred_min:
        pfs_predictions_scaled = (pfs_predictions - pred_min) / (pred_max - pred_min)
        pfs_predictions_scaled = pfs_predictions_scaled * (target_max - target_min) + target_min
    else:
        pfs_predictions_scaled = torch.full_like(pfs_predictions, all_targets.mean())
    
    mae = mean_absolute_error(all_targets.numpy(), pfs_predictions_scaled.numpy())
    
    # Spearman correlation
    spearman, _ = spearmanr(all_targets.numpy(), pfs_predictions_scaled.numpy())
    
    # C-index (concordance index) - MAIN METRIC for survival models
    c_index_fn = ConcordanceIndex()
    c_index = c_index_fn(all_risk_scores, all_targets, all_events)
    
    # Also calculate AUROC for binary classification (short vs long PFS)
    # Use dataset median PFS for consistency
    if dataset_median_pfs is not None:
        binary_labels = (all_targets.numpy() > dataset_median_pfs).astype(int)
    else:
        binary_labels = all_events.numpy().astype(int)
    
    try:
        # Use risk scores directly (higher risk = shorter PFS = class 0)
        # Invert for AUROC (we want to predict class 1 = long PFS)
        auroc = roc_auc_score(binary_labels, -all_risk_scores.numpy())
    except:
        auroc = 0.5
    
    return {
        'loss': avg_loss,
        'mae': mae,
        'spearman': spearman if not np.isnan(spearman) else 0.0,
        'c_index': c_index,
        'auroc': auroc
    }


def train_model(model, train_loader, val_loader, config, device):
    """Main training loop"""
    criterion = CoxPHLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    best_c_index = 0
    best_auroc = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_spearman': [],
        'val_c_index': [],
        'val_auroc': []
    }
    
    # Get dataset median PFS for consistent binary classification
    dataset_median_pfs = train_loader.dataset.median_pfs
    print(f"Dataset median PFS: {dataset_median_pfs:.2f} months")
    print(f"Binary classification: >={dataset_median_pfs:.2f} months = long PFS (1), <{dataset_median_pfs:.2f} = short PFS (0)")
    
    for epoch in range(config['num_epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, dataset_median_pfs)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_spearman'].append(val_metrics['spearman'])
        history['val_c_index'].append(val_metrics['c_index'])
        history['val_auroc'].append(val_metrics['auroc'])
        
        # Learning rate scheduling
        scheduler.step(val_metrics['c_index'])
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['num_epochs']}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"MAE: {val_metrics['mae']:.4f}, "
              f"Spearman: {val_metrics['spearman']:.4f}, "
              f"C-index: {val_metrics['c_index']:.4f}, "
              f"AUROC: {val_metrics['auroc']:.4f}")
        
        # Save best model (based on C-index, the standard metric for survival models)
        if val_metrics['c_index'] > best_c_index:
            best_c_index = val_metrics['c_index']
            best_auroc = val_metrics['auroc']
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_c_index': val_metrics['c_index'],
                'val_auroc': val_metrics['auroc'],
                'config': config
            }, config['save_path'])
            
            print(f"  >> Saved best model: C-index={val_metrics['c_index']:.4f}, AUROC={val_metrics['auroc']:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best C-index: {best_c_index:.4f}, Best AUROC: {best_auroc:.4f}")
            break
        
        # Check if we've reached target
        if val_metrics['auroc'] >= 0.70:
            print(f"\n** TARGET AUROC REACHED: {val_metrics['auroc']:.4f} >= 0.70 **")
            print(f"Success after {epoch+1} epochs!")
            break
    
    return history


def main():
    print("=" * 60)
    print("DeepSurv Model Training (Multimodal Survival Analysis)")
    print("=" * 60)
    
    # Configuration
    CONFIG = {
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'patience': 15,
        'save_path': 'artifacts/models/deepsurv_best.pth',
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.3
    }
    
    # Create save directory
    os.makedirs('artifacts/models', exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    X_genomic = np.load('artifacts/X_tabular_clean.npy')  # Genomic features
    X_drug = np.load('artifacts/X_drug_fp_clean.npy')     # Drug fingerprints
    X_clinical = np.load('artifacts/X_clinical_aligned.npy') # Clinical features (aligned)
    y = np.load('artifacts/y_clean.npy')                   # PFS targets
    
    # Load censoring information from original dataset
    df = pd.read_csv('data/processed/ml_dataset_complete.csv')
    pfs_status = df['pfs_status'].values[:len(y)]  # Event indicators (1=event, 0=censored)
    
    print(f"  Initial shapes - Genomic: {X_genomic.shape}, Drug: {X_drug.shape}, Clinical: {X_clinical.shape}, Targets: {y.shape}")
    print(f"  Event indicators: {pfs_status.shape}")
    print(f"  Events: {np.sum(pfs_status == 1)} progressions, {np.sum(pfs_status == 0)} censored")
    
    # Remove NaN rows
    mask = ~(np.isnan(X_genomic).any(axis=1) | 
             np.isnan(X_drug).any(axis=1) | 
             np.isnan(X_clinical).any(axis=1) | 
             np.isnan(y) |
             np.isnan(pfs_status))
    
    if not mask.all():
        print(f"  Found {(~mask).sum()} rows with NaN values, removing them...")
        X_genomic = X_genomic[mask]
        X_drug = X_drug[mask]
        X_clinical = X_clinical[mask]
        y = y[mask]
        pfs_status = pfs_status[mask]
    
    print(f"  Final shapes - Genomic: {X_genomic.shape}, Drug: {X_drug.shape}, Clinical: {X_clinical.shape}, Targets: {y.shape}")
    print(f"  PFS range: [{y.min():.2f}, {y.max():.2f}] months")
    print(f"  PFS mean: {y.mean():.2f}, median: {np.median(y):.2f}")
    print(f"  Final events: {np.sum(pfs_status == 1)} progressions ({100*np.sum(pfs_status == 1)/len(pfs_status):.1f}%), {np.sum(pfs_status == 0)} censored ({100*np.sum(pfs_status == 0)/len(pfs_status):.1f}%)")
    
    # Split data (70/15/15)
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    
    print(f"\nSplitting data...")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")
    
    # Create datasets
    train_dataset = MultimodalDataset(
        X_genomic[train_idx], X_drug[train_idx], X_clinical[train_idx], y[train_idx], pfs_status[train_idx]
    )
    val_dataset = MultimodalDataset(
        X_genomic[val_idx], X_drug[val_idx], X_clinical[val_idx], y[val_idx], pfs_status[val_idx]
    )
    test_dataset = MultimodalDataset(
        X_genomic[test_idx], X_drug[test_idx], X_clinical[test_idx], y[test_idx], pfs_status[test_idx]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Create model
    print("\nCreating DeepSurv model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    model = DeepSurv(
        genomic_dim=X_genomic.shape[1],
        drug_dim=X_drug.shape[1],
        clinical_dim=X_clinical.shape[1],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train
    print("\nStarting training...")
    print(f"Target: AUROC >= 0.70\n")
    
    history = train_model(model, train_loader, val_loader, CONFIG, device)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(CONFIG['save_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = CoxPHLoss()
    test_metrics = evaluate(model, test_loader, criterion, device, train_dataset.median_pfs)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  Spearman: {test_metrics['spearman']:.4f}")
    print(f"  C-index: {test_metrics['c_index']:.4f}")
    print(f"  AUROC: {test_metrics['auroc']:.4f}")
    
    if test_metrics['auroc'] >= 0.70:
        print(f"\n** SUCCESS! AUROC >= 0.70 achieved! **")
    else:
        print(f"\n** AUROC < 0.70. Consider:")
        print(f"   1. Implementing MOGONET (multi-omics integration)")
        print(f"   2. Adding contrastive pretraining")
        print(f"   3. Using ensemble methods")
        print(f"   See EVIDENCE_BASED_ARCHITECTURES.md for details")
    
    # Save final results
    results = {
        'test_metrics': test_metrics,
        'history': history,
        'config': CONFIG
    }
    
    import pickle
    with open('artifacts/deepsurv_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to artifacts/deepsurv_results.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()
