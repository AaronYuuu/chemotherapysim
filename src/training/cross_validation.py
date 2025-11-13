"""
Cross-Validation for Robust AUROC Estimate

Performs 5-fold stratified cross-validation to get a more reliable estimate
of model performance and reduce overfitting to a single validation set.
"""

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_absolute_error
from scipy.stats import spearmanr
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from training.train_ensemble import EnsemblePredictor
from models.deepsurv_model import ConcordanceIndex
from models.attention_model import train_attention_model


def train_fold(X_train, y_train, X_val, y_val, input_dim=150, fold_num=1):
    """Train ensemble on one fold."""
    ensemble = EnsemblePredictor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train models (quietly)
    ensemble.train_xgboost(X_train, y_train, X_val, y_val)
    ensemble.train_gradient_boosting(X_train, y_train, X_val, y_val)
    ensemble.train_random_forest(X_train, y_train, X_val, y_val)
    ensemble.train_ridge(X_train, y_train, X_val, y_val)
    
    # Train attention with unique save path per fold
    save_path = f'artifacts/models/attention_fold{fold_num}.pth'
    attention_model = train_attention_model(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        device=device,
        verbose=False,  # Quiet mode
        save_path=save_path
    )
    ensemble.models['attention'] = attention_model
    
    # Optimize weights
    ensemble.optimize_weights(X_val, y_val)
    
    return ensemble


def main():
    """Run 5-fold cross-validation."""
    print("=" * 60)
    print("5-Fold Cross-Validation")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    X = np.load('artifacts/X_selected.npy')
    y = np.load('artifacts/y_engineered.npy')
    pfs_status = np.load('artifacts/pfs_status_engineered.npy')
    
    print(f"  Features: {X.shape}")
    print(f"  Samples: {len(y)}")
    
    # Create binary target for stratification
    median_pfs = np.median(y)
    y_binary = (y > median_pfs).astype(int)
    
    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    
    print("\n" + "=" * 60)
    print("Training Folds")
    print("=" * 60)
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_binary), 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}/5")
        print(f"{'='*60}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        pfs_status_val = pfs_status[val_idx]
        
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples")
        
        # Train ensemble
        ensemble = train_fold(X_train, y_train, X_val, y_val, input_dim=X_train.shape[1], fold_num=fold_idx)
        
        # Evaluate
        val_pred = ensemble.predict(X_val)
        
        val_mae = mean_absolute_error(y_val, val_pred)
        val_spearman, _ = spearmanr(y_val, val_pred)
        
        median_fold = np.median(y_val)
        y_val_binary = (y_val > median_fold).astype(int)
        val_auroc = roc_auc_score(y_val_binary, val_pred)
        
        c_index_fn = ConcordanceIndex()
        risk_scores = -torch.FloatTensor(val_pred)
        times = torch.FloatTensor(y_val)
        events = torch.FloatTensor(pfs_status_val)
        val_c_index = c_index_fn(risk_scores, times, events)
        
        print(f"\nFold {fold_idx} Results:")
        print(f"  MAE: {val_mae:.4f}")
        print(f"  Spearman: {val_spearman:.4f}")
        print(f"  C-index: {val_c_index:.4f}")
        print(f"  AUROC: {val_auroc:.4f}")
        
        fold_results.append({
            'fold': fold_idx,
            'mae': val_mae,
            'spearman': val_spearman,
            'c_index': val_c_index,
            'auroc': val_auroc
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("Cross-Validation Summary")
    print("=" * 60)
    
    aurocs = [r['auroc'] for r in fold_results]
    maes = [r['mae'] for r in fold_results]
    spearmans = [r['spearman'] for r in fold_results]
    c_indices = [r['c_index'] for r in fold_results]
    
    print(f"\nAUROC across folds:")
    for i, auroc in enumerate(aurocs, 1):
        marker = "✓" if auroc >= 0.70 else " "
        print(f"  Fold {i}: {auroc:.4f} {marker}")
    
    print(f"\nMean ± Std:")
    print(f"  AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"  MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"  Spearman: {np.mean(spearmans):.4f} ± {np.std(spearmans):.4f}")
    print(f"  C-index: {np.mean(c_indices):.4f} ± {np.std(c_indices):.4f}")
    
    if np.mean(aurocs) >= 0.70:
        print(f"\n🎉 SUCCESS! Mean AUROC >= 0.70!")
    else:
        print(f"\n  Gap to 0.70: {0.70 - np.mean(aurocs):.4f}")
    
    # Save results
    cv_results = {
        'fold_results': fold_results,
        'mean_auroc': np.mean(aurocs),
        'std_auroc': np.std(aurocs),
        'mean_mae': np.mean(maes),
        'mean_spearman': np.mean(spearmans),
        'mean_c_index': np.mean(c_indices)
    }
    
    with open('artifacts/cv_results.pkl', 'wb') as f:
        pickle.dump(cv_results, f)
    
    print(f"\nResults saved: artifacts/cv_results.pkl")
    
    print("\n" + "=" * 60)
    print("Cross-validation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
