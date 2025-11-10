"""
Ensemble Model for Cancer PFS Prediction

Combines multiple model types:
1. DeepSurv (survival analysis with neural networks)
2. XGBoost (gradient boosting)
3. Random Forest (tree ensemble)
4. Ridge Regression (linear baseline)

Uses stacking with optimized weights.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from scipy.stats import spearmanr
from scipy.optimize import minimize
import pickle
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.deepsurv_model import DeepSurv, ConcordanceIndex
from models.attention_model import train_attention_model, predict_attention, get_feature_importance


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models.
    """
    def __init__(self):
        self.models = {}
        self.weights = None
        self.scaler = None
    
    def add_model(self, name, model):
        """Add a model to the ensemble"""
        self.models[name] = model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("\nTraining XGBoost...")
        from xgboost import XGBRegressor
        
        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.models['xgboost'] = model
        
        # Evaluate
        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_spearman, _ = spearmanr(y_val, val_pred)
        
        print(f"  XGBoost - Val MAE: {val_mae:.4f}, Spearman: {val_spearman:.4f}")
        
        return model
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest model"""
        print("\nTraining Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        # Evaluate
        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_spearman, _ = spearmanr(y_val, val_pred)
        
        print(f"  Random Forest - Val MAE: {val_mae:.4f}, Spearman: {val_spearman:.4f}")
        
        return model
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Train Gradient Boosting model"""
        print("\nTraining Gradient Boosting...")
        
        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        self.models['gradient_boosting'] = model
        
        # Evaluate
        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_spearman, _ = spearmanr(y_val, val_pred)
        
        print(f"  Gradient Boosting - Val MAE: {val_mae:.4f}, Spearman: {val_spearman:.4f}")
        
        return model
    
    def train_ridge(self, X_train, y_train, X_val, y_val):
        """Train Ridge regression (linear baseline)"""
        print("\nTraining Ridge Regression...")
        
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        self.models['ridge'] = model
        
        # Evaluate
        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_spearman, _ = spearmanr(y_val, val_pred)
        
        print(f"  Ridge - Val MAE: {val_mae:.4f}, Spearman: {val_spearman:.4f}")
        
        return model
    
    def get_predictions(self, X):
        """Get predictions from all models"""
        predictions = {}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for name, model in self.models.items():
            if name == 'deepsurv':
                # Handle DeepSurv separately (needs torch)
                continue
            elif name == 'attention':
                # Handle attention model
                predictions[name] = predict_attention(model, X, device=device)
            else:
                predictions[name] = model.predict(X)
        
        return predictions
    
    def optimize_weights(self, X_val, y_val):
        """
        Optimize ensemble weights using validation set.
        """
        print("\nOptimizing ensemble weights...")
        
        # Get predictions from all models
        predictions = self.get_predictions(X_val)
        
        # Stack predictions
        pred_matrix = np.column_stack(list(predictions.values()))
        
        # Objective: minimize MAE
        def objective(weights):
            weights = np.abs(weights)  # Ensure positive
            weights = weights / weights.sum()  # Normalize
            ensemble_pred = pred_matrix @ weights
            mae = mean_absolute_error(y_val, ensemble_pred)
            return mae
        
        # Initial weights (equal)
        n_models = len(predictions)
        initial_weights = np.ones(n_models) / n_models
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        # Get optimized weights
        weights = np.abs(result.x)
        weights = weights / weights.sum()
        
        self.weights = dict(zip(predictions.keys(), weights))
        
        # Print weights
        print("\nOptimized weights:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f}")
        
        # Evaluate ensemble
        ensemble_pred = pred_matrix @ weights
        ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
        ensemble_spearman, _ = spearmanr(y_val, ensemble_pred)
        
        # Calculate AUROC
        median_pfs = np.median(y_val)
        y_binary = (y_val > median_pfs).astype(int)
        ensemble_auroc = roc_auc_score(y_binary, ensemble_pred)
        
        print(f"\nEnsemble Performance:")
        print(f"  MAE: {ensemble_mae:.4f}")
        print(f"  Spearman: {ensemble_spearman:.4f}")
        print(f"  AUROC: {ensemble_auroc:.4f}")
        
        return weights
    
    def predict(self, X):
        """Make ensemble prediction"""
        predictions = self.get_predictions(X)
        
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred
        
        return ensemble_pred
    
    def save(self, path):
        """Save ensemble model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'weights': self.weights
            }, f)
        print(f"\nEnsemble saved to: {path}")


def train_ensemble(X_train, y_train, X_val, y_val, X_test, y_test, pfs_status_test):
    """
    Train ensemble of models.
    """
    print("=" * 60)
    print("Training Ensemble of Models (with Attention)")
    print("=" * 60)
    
    ensemble = EnsemblePredictor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train individual models
    try:
        ensemble.train_xgboost(X_train, y_train, X_val, y_val)
    except Exception as e:
        print(f"XGBoost training failed: {e}")
    
    ensemble.train_gradient_boosting(X_train, y_train, X_val, y_val)
    ensemble.train_random_forest(X_train, y_train, X_val, y_val)
    ensemble.train_ridge(X_train, y_train, X_val, y_val)
    
    # Train attention model
    attention_model = train_attention_model(
        X_train, y_train, X_val, y_val,
        input_dim=X_train.shape[1],
        device=device
    )
    ensemble.models['attention'] = attention_model
    
    # Optimize weights
    weights = ensemble.optimize_weights(X_val, y_val)
    
    # Test set evaluation
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    
    test_pred = ensemble.predict(X_test)
    
    test_mae = mean_absolute_error(y_test, test_pred)
    test_spearman, _ = spearmanr(y_test, test_pred)
    
    # AUROC
    median_pfs = np.median(y_test)
    y_binary = (y_test > median_pfs).astype(int)
    test_auroc = roc_auc_score(y_binary, test_pred)
    
    # C-index using actual censoring info
    c_index_fn = ConcordanceIndex()
    # Convert predictions to risk scores (invert since higher PFS = lower risk)
    risk_scores = -torch.FloatTensor(test_pred)
    times = torch.FloatTensor(y_test)
    events = torch.FloatTensor(pfs_status_test)
    test_c_index = c_index_fn(risk_scores, times, events)
    
    print(f"\nEnsemble Test Results:")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  Spearman: {test_spearman:.4f}")
    print(f"  C-index: {test_c_index:.4f}")
    print(f"  AUROC: {test_auroc:.4f}")
    
    if test_auroc >= 0.70:
        print(f"\n** SUCCESS! AUROC >= 0.70 achieved! **")
    else:
        print(f"\n** AUROC = {test_auroc:.4f} (target: 0.70)")
        print(f"   Gap to target: {0.70 - test_auroc:.4f}")
    
    # Individual model performance on test set
    print("\nIndividual Model Test Performance:")
    test_predictions = ensemble.get_predictions(X_test)
    for name, pred in test_predictions.items():
        mae = mean_absolute_error(y_test, pred)
        spearman, _ = spearmanr(y_test, pred)
        auroc = roc_auc_score(y_binary, pred)
        print(f"  {name}: MAE={mae:.4f}, Spearman={spearman:.4f}, AUROC={auroc:.4f}")
    
    # Show attention-based feature importance
    if 'attention' in ensemble.models:
        print("\nAnalyzing feature importance from attention model...")
        # Load feature names
        try:
            with open('artifacts/feature_engineering_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            feature_names = metadata.get('feature_names', [])
            
            if len(feature_names) > 0:
                importance = get_feature_importance(
                    ensemble.models['attention'], 
                    X_test, 
                    feature_names,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                )
            else:
                print("  Feature names not found in metadata")
        except Exception as e:
            print(f"  Could not analyze feature importance: {e}")
    
    return ensemble, test_pred


def main():
    """Main execution"""
    print("=" * 60)
    print("Ensemble Model Training with Engineered Features")
    print("=" * 60)
    
    # Load engineered features
    print("\nLoading engineered features...")
    X = np.load('artifacts/X_engineered.npy')
    y = np.load('artifacts/y_engineered.npy')
    pfs_status = np.load('artifacts/pfs_status_engineered.npy')
    
    print(f"  Features: {X.shape}")
    print(f"  Targets: {y.shape}")
    print(f"  Events: {np.sum(pfs_status == 1)} progressions, {np.sum(pfs_status == 0)} censored")
    
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
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    pfs_status_test = pfs_status[test_idx]
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")
    
    # Train ensemble
    ensemble, test_pred = train_ensemble(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        pfs_status_test
    )
    
    # Save ensemble
    ensemble.save('artifacts/models/ensemble_model.pkl')
    
    # Save results
    results = {
        'test_predictions': test_pred,
        'test_targets': y_test,
        'test_pfs_status': pfs_status_test,
        'weights': ensemble.weights
    }
    
    with open('artifacts/ensemble_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "=" * 60)
    print("Ensemble training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
