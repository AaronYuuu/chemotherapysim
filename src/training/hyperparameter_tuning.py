"""
Hyperparameter Tuning for Attention Model + Ensemble

Uses Optuna for Bayesian optimization of:
- Attention model architecture (hidden_dim, num_heads, num_layers, dropout)
- Training hyperparameters (learning_rate, batch_size)
- Ensemble composition
"""

import numpy as np
import torch
import optuna
from sklearn.metrics import roc_auc_score, mean_absolute_error
from scipy.stats import spearmanr
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from training.train_ensemble import EnsemblePredictor
from models.attention_model import train_attention_model


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function.
    """
    # Hyperparameters to tune
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
    num_layers = trial.suggest_int('num_layers', 2, 5)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # XGBoost params
    xgb_max_depth = trial.suggest_int('xgb_max_depth', 4, 10)
    xgb_learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.1)
    
    # Gradient Boosting params
    gb_max_depth = trial.suggest_int('gb_max_depth', 3, 8)
    gb_learning_rate = trial.suggest_float('gb_learning_rate', 0.01, 0.1)
    
    try:
        # Train ensemble with these hyperparameters
        ensemble = EnsemblePredictor()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # XGBoost
        from xgboost import XGBRegressor
        xgb_model = XGBRegressor(
            n_estimators=500,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='mae'
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        ensemble.models['xgboost'] = xgb_model
        
        # Gradient Boosting
        from sklearn.ensemble import GradientBoostingRegressor
        gb_model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=gb_max_depth,
            learning_rate=gb_learning_rate,
            subsample=0.8,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        ensemble.models['gradient_boosting'] = gb_model
        
        # Ridge (simple, no tuning needed)
        from sklearn.linear_model import Ridge
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        ensemble.models['ridge'] = ridge_model
        
        # Random Forest (simple, no tuning)
        from sklearn.ensemble import RandomForestRegressor
        rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        ensemble.models['random_forest'] = rf_model
        
        # Attention model with tuned hyperparameters
        attention_model = train_attention_model(
            X_train, y_train, X_val, y_val,
            input_dim=X_train.shape[1],
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=device,
            verbose=False
        )
        ensemble.models['attention'] = attention_model
        
        # Optimize weights
        ensemble.optimize_weights(X_val, y_val)
        
        # Evaluate
        val_pred = ensemble.predict(X_val)
        
        median_pfs = np.median(y_val)
        y_binary = (y_val > median_pfs).astype(int)
        auroc = roc_auc_score(y_binary, val_pred)
        
        return auroc
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0


def main():
    """Run hyperparameter optimization."""
    print("=" * 60)
    print("Hyperparameter Tuning with Optuna")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    X = np.load('artifacts/X_selected.npy')
    y = np.load('artifacts/y_engineered.npy')
    
    # Split
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_size = int(0.8 * n)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    
    # Create study with persistence
    print("\nStarting optimization...")
    print("  Objective: Maximize AUROC")
    print("  Trials: 50")
    print("  Method: TPE (Tree-structured Parzen Estimator)")
    
    study_name = "pfs_ensemble_optimization"
    storage_name = f"sqlite:///artifacts/{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True  # Resume if interrupted
    )
    
    print(f"  Study will be saved to: {storage_name}")
    print(f"  Current trials completed: {len(study.trials)}")
    
    # Optimize with exception handling
    try:
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val),
            n_trials=50,
            show_progress_bar=True,
            catch=(Exception,)  # Continue on errors
        )
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        print(f"Completed {len(study.trials)} trials so far.")
    
    # Results
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    
    if len(study.trials) == 0:
        print("No trials completed.")
        return
    
    # Get completed trials only
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) == 0:
        print("No trials completed successfully.")
        return
    
    print(f"\nCompleted trials: {len(completed_trials)}/{len(study.trials)}")
    print(f"Best AUROC: {study.best_value:.4f}")
    
    if study.best_value >= 0.70:
        print(f"🎉 SUCCESS! AUROC >= 0.70 achieved!")
    else:
        print(f"  Gap to 0.70: {0.70 - study.best_value:.4f}")
    
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save
    with open('artifacts/optuna_study.pkl', 'wb') as f:
        pickle.dump(study, f)
    
    print(f"\nStudy saved: artifacts/optuna_study.pkl")
    
    # Top 5 trials
    print(f"\nTop 5 trials:")
    df = study.trials_dataframe()
    df_sorted = df.sort_values('value', ascending=False).head(5)
    
    for i, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"  {i}. Trial {row['number']}: AUROC={row['value']:.4f}")
    
    print("\n" + "=" * 60)
    print("Hyperparameter tuning complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
