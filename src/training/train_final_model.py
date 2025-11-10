"""
Train final ensemble with best hyperparameters from Optuna optimization.
"""

import numpy as np
import torch
import pickle
from sklearn.metrics import mean_absolute_error, roc_auc_score
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.deepsurv_model import ConcordanceIndex
from models.attention_model import train_attention_model, predict_attention
from training.train_ensemble import EnsemblePredictor


def main():
    """Train final model with best hyperparameters."""
    print("=" * 60)
    print("FINAL MODEL TRAINING - Best Hyperparameters")
    print("=" * 60)
    
    # Load best hyperparameters
    print("\nLoading best hyperparameters...")
    with open('artifacts/best_hyperparameters.pkl', 'rb') as f:
        best_config = pickle.load(f)
    
    print(f"  Best AUROC from tuning: {best_config['auroc']:.4f}")
    print(f"  Trial number: {best_config['trial_number']}")
    
    # Load data
    print("\nLoading data...")
    X = np.load('artifacts/X_selected.npy')
    y = np.load('artifacts/y_engineered.npy')
    pfs_status = np.load('artifacts/pfs_status_engineered.npy')
    
    print(f"  Features: {X.shape}")
    print(f"  Samples: {len(y)}")
    
    # Split data
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    pfs_status_test = pfs_status[test_idx]
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Extract hyperparameters
    params = best_config['params']
    
    print("\n" + "=" * 60)
    print("Training Ensemble with Best Hyperparameters")
    print("=" * 60)
    
    ensemble = EnsemblePredictor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    print(f"  max_depth={params['xgb_max_depth']}, learning_rate={params['xgb_learning_rate']:.4f}")
    xgb_model = XGBRegressor(
        n_estimators=500,
        max_depth=params['xgb_max_depth'],
        learning_rate=params['xgb_learning_rate'],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='mae'
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    ensemble.models['xgboost'] = xgb_model
    
    val_pred = xgb_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_spearman, _ = spearmanr(y_val, val_pred)
    print(f"  XGBoost - Val MAE: {val_mae:.4f}, Spearman: {val_spearman:.4f}")
    
    # Train Gradient Boosting
    print("\nTraining Gradient Boosting...")
    print(f"  max_depth={params['gb_max_depth']}, learning_rate={params['gb_learning_rate']:.4f}")
    gb_model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=params['gb_max_depth'],
        learning_rate=params['gb_learning_rate'],
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    ensemble.models['gradient_boosting'] = gb_model
    
    val_pred = gb_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_spearman, _ = spearmanr(y_val, val_pred)
    print(f"  Gradient Boosting - Val MAE: {val_mae:.4f}, Spearman: {val_spearman:.4f}")
    
    # Train Random Forest (using default params)
    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    ensemble.models['random_forest'] = rf_model
    
    val_pred = rf_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_spearman, _ = spearmanr(y_val, val_pred)
    print(f"  Random Forest - Val MAE: {val_mae:.4f}, Spearman: {val_spearman:.4f}")
    
    # Train Ridge (using default params)
    print("\nTraining Ridge Regression...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ensemble.models['ridge'] = ridge_model
    
    val_pred = ridge_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_spearman, _ = spearmanr(y_val, val_pred)
    print(f"  Ridge - Val MAE: {val_mae:.4f}, Spearman: {val_spearman:.4f}")
    
    # Train Attention Model
    print("\nTraining Attention Model with best hyperparameters...")
    print(f"  hidden_dim={params['hidden_dim']}, num_heads={params['num_heads']}, num_layers={params['num_layers']}")
    print(f"  dropout={params['dropout']:.4f}, learning_rate={params['learning_rate']:.6f}, batch_size={params['batch_size']}")
    
    attention_model = train_attention_model(
        X_train, y_train, X_val, y_val,
        input_dim=150,
        hidden_dim=params['hidden_dim'],
        num_heads=params['num_heads'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        learning_rate=params['learning_rate'],
        batch_size=params['batch_size'],
        device=device,
        verbose=True,
        save_path='artifacts/models/attention_final_best.pth'
    )
    ensemble.models['attention'] = attention_model
    
    val_pred = predict_attention(attention_model, X_val, device=device)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_spearman, _ = spearmanr(y_val, val_pred)
    print(f"  Attention - Val MAE: {val_mae:.4f}, Spearman: {val_spearman:.4f}")
    
    # Optimize ensemble weights
    print("\n" + "=" * 60)
    print("Optimizing Ensemble Weights")
    print("=" * 60)
    
    ensemble.optimize_weights(X_val, y_val)
    
    print("\nOptimized weights:")
    for name, weight in ensemble.weights.items():
        print(f"  {name}: {weight:.4f}")
    
    # Validate
    val_pred = ensemble.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_spearman, _ = spearmanr(y_val, val_pred)
    
    median_pfs = np.median(y_val)
    y_binary = (y_val > median_pfs).astype(int)
    val_auroc = roc_auc_score(y_binary, val_pred)
    
    print(f"\nEnsemble Validation Performance:")
    print(f"  MAE: {val_mae:.4f}")
    print(f"  Spearman: {val_spearman:.4f}")
    print(f"  AUROC: {val_auroc:.4f}")
    
    # Test
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    
    test_pred = ensemble.predict(X_test)
    
    test_mae = mean_absolute_error(y_test, test_pred)
    test_spearman, _ = spearmanr(y_test, test_pred)
    
    median_pfs = np.median(y_test)
    y_binary = (y_test > median_pfs).astype(int)
    test_auroc = roc_auc_score(y_binary, test_pred)
    
    c_index_fn = ConcordanceIndex()
    risk_scores = -torch.FloatTensor(test_pred)
    times = torch.FloatTensor(y_test)
    events = torch.FloatTensor(pfs_status_test)
    test_c_index = c_index_fn(risk_scores, times, events)
    
    print(f"\nFinal Test Results (with best hyperparameters):")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  Spearman: {test_spearman:.4f}")
    print(f"  C-index: {test_c_index:.4f}")
    print(f"  **AUROC: {test_auroc:.4f}**")
    
    if test_auroc >= 0.70:
        print(f"\n🎉 SUCCESS! Test AUROC >= 0.70 achieved!")
        print(f"   Exceeds target by: {test_auroc - 0.70:.4f}")
    else:
        print(f"\n  Gap to 0.70: {0.70 - test_auroc:.4f}")
    
    # Individual model test performance
    print(f"\nIndividual Model Test Performance:")
    for name, model in ensemble.models.items():
        if name == 'attention':
            pred = predict_attention(model, X_test, device=device)
        else:
            pred = model.predict(X_test)
        
        auroc = roc_auc_score(y_binary, pred)
        mae = mean_absolute_error(y_test, pred)
        spearman, _ = spearmanr(y_test, pred)
        print(f"  {name}: AUROC={auroc:.4f}, MAE={mae:.4f}, Spearman={spearman:.4f}")
    
    # Save final model
    final_model = {
        'models': ensemble.models,
        'weights': ensemble.weights,
        'hyperparameters': params,
        'feature_dim': 150,
        'test_auroc': test_auroc,
        'test_c_index': test_c_index
    }
    
    with open('artifacts/models/final_optimized_ensemble.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    
    print(f"\nFinal model saved: artifacts/models/final_optimized_ensemble.pkl")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
