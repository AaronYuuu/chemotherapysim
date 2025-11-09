"""
Training script for GNN-based cancer drug response prediction model

This script trains the advanced GNN model with:
- Pathway-aware architecture
- Multi-task learning (PFS + resistance + pathway activity)
- Cross-attention mechanisms
- Compatible with existing infrastructure

Author: Aaron Yu
Date: November 8, 2025
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
import pandas as pd
import pickle

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.models.gnn_model import PathwayAwareGNN
from src.models.deep_learning import create_optimizer, create_scheduler
from src.data.pathway_utils import PathwayGraphBuilder

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Paths
DATA_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_stratified" / "gnn_advanced"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Configuration
CONFIG = {
    # Data split
    "train_size": 0.70,
    "val_size": 0.15,
    "test_size": 0.15,
    
    # Model architecture
    "genomic_dim": 1318,
    "drug_fp_dim": 8192,
    "embed_dim": 256,
    "num_resistance_classes": 5,
    "num_pathways": 50,
    "use_pathway_graph": False,  # Start with fallback mode
    "dropout_genomic": 0.4,
    "dropout_drug": 0.4,
    "dropout_head": 0.3,
    
    # Training
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-3,
    "warmup_epochs": 10,
    
    # Multi-task learning weights
    "pfs_weight": 1.0,
    "resistance_weight": 0.0,  # Disabled for now (no labels)
    "pathway_weight": 0.0,     # Disabled for now (no labels)
    
    # Early stopping
    "patience": 15,
    "min_delta": 1e-4,
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "random_seed": RANDOM_SEED
}


class DrugResponseDataset(Dataset):
    """Dataset for drug response prediction"""
    def __init__(self, genomic_features, drug_features, targets):
        self.genomic_features = torch.FloatTensor(genomic_features)
        self.drug_features = torch.FloatTensor(drug_features)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'genomic': self.genomic_features[idx],
            'drug': self.drug_features[idx],
            'target': self.targets[idx]
        }


def load_data():
    """Load preprocessed data"""
    print("\nLoading data...")
    
    # Load features
    X_genomic = np.load(DATA_DIR / "X_tabular_clean.npy")
    X_drug = np.load(DATA_DIR / "X_drug_fp_clean.npy")
    y = np.load(DATA_DIR / "y_pfs_clean.npy")
    
    # Ensure consistent shapes
    min_len = min(len(X_genomic), len(X_drug), len(y))
    X_genomic = X_genomic[:min_len]
    X_drug = X_drug[:min_len]
    y = y[:min_len]
    
    print(f"  Genomic features: {X_genomic.shape}")
    print(f"  Drug features: {X_drug.shape}")
    print(f"  Targets: {y.shape}")
    print(f"  PFS range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X_genomic, X_drug, y


def split_data(X_genomic, X_drug, y, config):
    """Split data with stratification"""
    print("\nSplitting data...")
    
    # Create quantiles for stratification
    n_quantiles = 5
    quantiles = pd.qcut(y, q=n_quantiles, labels=False, duplicates='drop')
    
    # First split: train vs (val + test)
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(config['val_size'] + config['test_size']),
        stratify=quantiles,
        random_state=config['random_seed']
    )
    
    # Second split: val vs test
    temp_quantiles = quantiles[temp_idx]
    val_size_adjusted = config['val_size'] / (config['val_size'] + config['test_size'])
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_size_adjusted),
        stratify=temp_quantiles,
        random_state=config['random_seed']
    )
    
    # Create datasets
    train_dataset = DrugResponseDataset(
        X_genomic[train_idx], X_drug[train_idx], y[train_idx]
    )
    val_dataset = DrugResponseDataset(
        X_genomic[val_idx], X_drug[val_idx], y[val_idx]
    )
    test_dataset = DrugResponseDataset(
        X_genomic[test_idx], X_drug[test_idx], y[test_idx]
    )
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset


def train_epoch(model, dataloader, optimizer, criterion, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        genomic = batch['genomic'].to(device)
        drug = batch['drug'].to(device)
        target = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if config['resistance_weight'] > 0 or config['pathway_weight'] > 0:
            pred, additional = model(genomic, drug, return_all_tasks=True)
            
            # Multi-task loss
            pfs_loss = criterion(pred, target)
            loss = config['pfs_weight'] * pfs_loss
            
            # Add other task losses if needed
            # (resistance and pathway losses would go here)
        else:
            # Single task (PFS only)
            pred = model(genomic, drug)
            loss = criterion(pred, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, config):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            genomic = batch['genomic'].to(device)
            drug = batch['drug'].to(device)
            target = batch['target'].to(device)
            
            pred = model(genomic, drug)
            loss = criterion(pred, target)
            
            total_loss += loss.item()
            predictions.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(predictions, targets)
    
    return {
        'loss': total_loss / len(dataloader),
        'mse': mse,
        'mae': mae,
        'spearman': spearman_corr
    }


def train_model(model, train_loader, val_loader, config):
    """Main training loop"""
    device = config['device']
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = create_optimizer(model, config['learning_rate'], config['weight_decay'])
    scheduler = create_scheduler(
        optimizer, 
        config['warmup_epochs'], 
        config['num_epochs']
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_spearman': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nTraining on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(config['num_epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, config)
        
        # Update learning rate
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_spearman'].append(val_metrics['spearman'])
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val MAE: {val_metrics['mae']:.4f}")
            print(f"  Val Spearman: {val_metrics['spearman']:.4f}")
        
        # Early stopping
        if val_metrics['loss'] < (best_val_loss - config['min_delta']):
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            # Save best model
            model.save_checkpoint(
                str(CHECKPOINT_DIR / "best_model.pt"),
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics
            )
        else:
            patience_counter += 1
            
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    return history


def plot_training_history(history):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(history['val_mae'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Validation MAE')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Spearman correlation
    axes[1, 0].plot(history['val_spearman'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Spearman Correlation')
    axes[1, 0].set_title('Validation Spearman Correlation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Hide last subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved to {CHECKPOINT_DIR / 'training_curves.png'}")


def main():
    """Main training function"""
    print("=" * 60)
    print("GNN Model Training")
    print("=" * 60)
    
    # Load data
    X_genomic, X_drug, y = load_data()
    
    # Split data
    train_dataset, val_dataset, test_dataset = split_data(X_genomic, X_drug, y, CONFIG)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False
    )
    
    # Create model
    print("\nCreating GNN model...")
    model = PathwayAwareGNN(**{k: v for k, v in CONFIG.items() if k in [
        'genomic_dim', 'drug_fp_dim', 'embed_dim', 'num_resistance_classes',
        'num_pathways', 'use_pathway_graph', 'dropout_genomic', 'dropout_drug',
        'dropout_head'
    ]})
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save config
    with open(CHECKPOINT_DIR / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Train
    history = train_model(model, train_loader, val_loader, CONFIG)
    
    # Plot training curves
    plot_training_history(history)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    best_model = PathwayAwareGNN.from_checkpoint(
        str(CHECKPOINT_DIR / "best_model.pt"),
        device=CONFIG['device']
    )
    
    criterion = nn.MSELoss()
    test_metrics = evaluate(best_model, test_loader, criterion, CONFIG['device'], CONFIG)
    
    print(f"\nTest Metrics:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  MSE: {test_metrics['mse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  Spearman: {test_metrics['spearman']:.4f}")
    
    # Save test metrics
    with open(CHECKPOINT_DIR / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: {CHECKPOINT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
