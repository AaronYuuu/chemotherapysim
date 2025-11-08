"""
XGBoost Training for First Treatment Stratum
Optimized for small datasets (N < 1000)
Features:
  - K-fold cross-validation (K=5)
  - Bayesian hyperparameter optimization
  - Feature selection using XGBoost importance
  - Conservative regularization
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
import json
from pathlib import Path
import pickle
import warnings
import sys
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_and_split_data():
    """Load first_treatment data and split patient-level"""
    print("=" * 70)
    print("LOADING DATA FOR FIRST TREATMENT (XGBoost)")
    print("=" * 70)
    
    # Load full dataset
    X_tabular = np.load(PROJECT_ROOT / 'artifacts/X_tabular_clean.npy')
    X_drug_fp = np.load(PROJECT_ROOT / 'artifacts/X_drug_fp_clean.npy')
    y = np.load(PROJECT_ROOT / 'artifacts/y_clean.npy')
    patient_ids = np.load(PROJECT_ROOT / 'artifacts/patient_ids_clean.npy', allow_pickle=True)
    
    # Filter for first treatment
    treatment_lines = X_tabular[:, 0]
    mask = treatment_lines == 1.0
    
    X_tab = X_tabular[mask]
    X_drug = X_drug_fp[mask]
    y_filtered = y[mask]
    patients = patient_ids[mask]
    
    # Combine features
    X = np.hstack([X_tab, X_drug])
    
    print(f"\nFirst Treatment Dataset:")
    print(f"  Samples: {len(y_filtered)}")
    print(f"  Features: {X.shape[1]:,} (tabular: {X_tab.shape[1]:,}, drug: {X_drug.shape[1]:,})")
    print(f"  Unique patients: {len(np.unique(patients))}")
    print(f"  Target range: [{y_filtered.min():.3f}, {y_filtered.max():.3f}]")
    print(f"  Target mean: {y_filtered.mean():.3f} ± {y_filtered.std():.3f}")
    
    # Handle NaNs with median imputation
    print("\nHandling missing values...")
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"  Found {nan_count:,} NaNs, imputing with feature-wise medians")
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = col_medians[i]
    
    # Patient-level split (70/15/15)
    unique_patients = np.unique(patients)
    train_patients, temp_patients = train_test_split(
        unique_patients, test_size=0.3, random_state=42
    )
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.5, random_state=42
    )
    
    train_mask = np.isin(patients, train_patients)
    val_mask = np.isin(patients, val_patients)
    test_mask = np.isin(patients, test_patients)
    
    X_train, y_train = X[train_mask], y_filtered[train_mask]
    X_val, y_val = X[val_mask], y_filtered[val_mask]
    X_test, y_test = X[test_mask], y_filtered[test_mask]
    
    print(f"\nPatient-level splits:")
    print(f"  Training:   {len(train_patients):3d} patients ({len(y_train):3d} samples)")
    print(f"  Validation: {len(val_patients):3d} patients ({len(y_val):3d} samples)")
    print(f"  Test:       {len(test_patients):3d} patients ({len(y_test):3d} samples)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def feature_selection(X_train, y_train, n_features_target=500):
    """
    Aggressive feature selection using XGBoost importance.
    Reduces feature space from ~9500 to ~500 to stabilize model.
    """
    print("\n" + "=" * 70)
    print("FEATURE SELECTION")
    print("=" * 70)
    
    print(f"\nInitial features: {X_train.shape[1]:,}")
    print(f"Target features: {n_features_target}")
    
    # Train a preliminary XGBoost model to get feature importance
    print("\nTraining preliminary model for feature importance...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    }
    
    prelim_model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        verbose_eval=False
    )
    
    # Get feature importance scores
    importance = prelim_model.get_score(importance_type='gain')
    
    # Create importance array (0 for features not in model)
    importance_array = np.zeros(X_train.shape[1])
    for feat_name, score in importance.items():
        feat_idx = int(feat_name.replace('f', ''))
        importance_array[feat_idx] = score
    
    # Select top features
    top_indices = np.argsort(importance_array)[-n_features_target:]
    top_indices = np.sort(top_indices)  # Keep original order
    
    print(f"\nSelected {len(top_indices)} features based on gain importance")
    print(f"Importance range: [{importance_array[top_indices].min():.2f}, {importance_array[top_indices].max():.2f}]")
    print(f"Feature reduction: {X_train.shape[1]:,} -> {len(top_indices)} ({100*len(top_indices)/X_train.shape[1]:.1f}%)")
    
    return top_indices, prelim_model


def hyperparameter_search_cv(X_train, y_train, selected_features, n_folds=5):
    """
    Hyperparameter tuning with K-fold cross-validation.
    Tests multiple configurations and selects best based on CV AUROC.
    """
    print("\n" + "=" * 70)
    print(f"HYPERPARAMETER TUNING WITH {n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 70)
    
    # Select features
    X_selected = X_train[:, selected_features]
    
    # Define search space (conservative for small dataset)
    param_grid = [
        # Very conservative (prefer this for N=822)
        {'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 100, 'min_child_weight': 5, 
         'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 2.0},
        
        {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 150, 'min_child_weight': 5,
         'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 1.5},
        
        # Moderately conservative
        {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 100, 'min_child_weight': 3,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.0},
        
        {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 150, 'min_child_weight': 3,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.3, 'reg_lambda': 1.0},
        
        # Less conservative (test if more complexity helps)
        {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 100, 'min_child_weight': 2,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.3, 'reg_lambda': 0.5},
    ]
    
    print(f"\nTesting {len(param_grid)} hyperparameter configurations...")
    print(f"Each will be evaluated with {n_folds}-fold CV")
    
    # K-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    best_params = None
    best_cv_auroc = 0
    all_results = []
    
    threshold = np.log1p(6.0)  # 6-month threshold in log space
    
    for idx, params in enumerate(param_grid):
        print(f"\n[{idx+1}/{len(param_grid)}] Testing: max_depth={params['max_depth']}, "
              f"lr={params['learning_rate']}, n_est={params['n_estimators']}")
        
        cv_aurocs = []
        cv_rmses = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected)):
            X_fold_train, X_fold_val = X_selected[train_idx], X_selected[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Train model
            dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
            dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
            
            xgb_params = {
                'objective': 'reg:squarederror',
                'max_depth': params['max_depth'],
                'learning_rate': params['learning_rate'],
                'subsample': params['subsample'],
                'colsample_bytree': params['colsample_bytree'],
                'min_child_weight': params['min_child_weight'],
                'reg_alpha': params['reg_alpha'],
                'reg_lambda': params['reg_lambda'],
                'random_state': 42,
                'verbosity': 0
            }
            
            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=params['n_estimators'],
                evals=[(dval, 'val')],
                verbose_eval=False
            )
            
            # Evaluate
            y_val_pred = model.predict(dval)
            y_val_binary = (y_fold_val >= threshold).astype(int)
            
            # Only compute AUROC if both classes present
            if len(np.unique(y_val_binary)) > 1:
                auroc = roc_auc_score(y_val_binary, y_val_pred)
                cv_aurocs.append(auroc)
            
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_val_pred))
            cv_rmses.append(rmse)
        
        # Average across folds
        mean_auroc = np.mean(cv_aurocs) if cv_aurocs else 0
        std_auroc = np.std(cv_aurocs) if cv_aurocs else 0
        mean_rmse = np.mean(cv_rmses)
        
        print(f"  CV AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
        print(f"  CV RMSE:  {mean_rmse:.4f}")
        
        all_results.append({
            'params': params,
            'cv_auroc_mean': mean_auroc,
            'cv_auroc_std': std_auroc,
            'cv_rmse_mean': mean_rmse
        })
        
        # Track best
        if mean_auroc > best_cv_auroc:
            best_cv_auroc = mean_auroc
            best_params = params
    
    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETERS")
    print("=" * 70)
    print(f"CV AUROC: {best_cv_auroc:.4f}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    return best_params, all_results


def train_final_model(X_train, y_train, X_val, y_val, selected_features, best_params):
    """Train final model on full training set with best hyperparameters"""
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)
    
    # Select features
    X_train_selected = X_train[:, selected_features]
    X_val_selected = X_val[:, selected_features]
    
    print(f"\nUsing {len(selected_features)} selected features")
    print("Best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train_selected, label=y_train)
    dval = xgb.DMatrix(X_val_selected, label=y_val)
    
    # XGBoost parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': best_params['max_depth'],
        'learning_rate': best_params['learning_rate'],
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree'],
        'min_child_weight': best_params['min_child_weight'],
        'reg_alpha': best_params['reg_alpha'],
        'reg_lambda': best_params['reg_lambda'],
        'random_state': 42,
        'verbosity': 1
    }
    
    # Train with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}
    
    print("\nTraining with early stopping (patience=30)...")
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=best_params['n_estimators'] * 2,  # Allow more rounds with early stopping
        evals=evals,
        early_stopping_rounds=30,
        evals_result=evals_result,
        verbose_eval=20
    )
    
    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best validation RMSE: {model.best_score:.4f}")
    
    return model, evals_result


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, selected_features):
    """Evaluate model performance"""
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    # Select features
    X_train_selected = X_train[:, selected_features]
    X_val_selected = X_val[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train_selected)
    dval = xgb.DMatrix(X_val_selected)
    dtest = xgb.DMatrix(X_test_selected)
    
    # Make predictions
    y_train_pred = model.predict(dtrain)
    y_val_pred = model.predict(dval)
    y_test_pred = model.predict(dtest)
    
    # Regression metrics
    metrics = {}
    for split_name, y_true, y_pred in [
        ('train', y_train, y_train_pred),
        ('val', y_val, y_val_pred),
        ('test', y_test, y_test_pred)
    ]:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        metrics[split_name] = {'rmse': float(rmse), 'r2': float(r2)}
        print(f"\n{split_name.upper()} SET:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
    
    # Classification metrics at 6-month threshold
    print("\n" + "=" * 70)
    print("CLASSIFICATION METRICS (6-MONTH THRESHOLD)")
    print("=" * 70)
    
    threshold = np.log1p(6.0)  # log(6 + 1) to match training target
    
    classification_metrics = {}
    for split_name, y_true, y_pred in [
        ('train', y_train, y_train_pred),
        ('val', y_val, y_val_pred),
        ('test', y_test, y_test_pred)
    ]:
        y_true_binary = (y_true >= threshold).astype(int)
        
        if len(np.unique(y_true_binary)) > 1:  # Only compute AUROC if both classes present
            auroc = roc_auc_score(y_true_binary, y_pred)
            n_positive = y_true_binary.sum()
            n_negative = len(y_true_binary) - n_positive
            
            classification_metrics[split_name] = {
                'auroc': float(auroc),
                'n_positive': int(n_positive),
                'n_negative': int(n_negative),
                'threshold_months': 6.0
            }
            
            print(f"{split_name.upper()}: AUROC={auroc:.4f}, N+={n_positive}, N-={n_negative}")
        else:
            print(f"{split_name.upper()}: Cannot compute AUROC (only one class)")
    
    return metrics, classification_metrics


def save_model(model, metrics, classification_metrics, selected_features, cv_results, best_params):
    """Save model and metrics"""
    output_dir = PROJECT_ROOT / 'checkpoints_stratified' / 'first_treatment_xgboost'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save_model(str(output_dir / 'xgboost_model.json'))
    with open(output_dir / 'xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save selected features
    np.save(output_dir / 'selected_features.npy', selected_features)
    
    # Save metrics
    with open(output_dir / 'regression_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Format classification metrics for compatibility with deep learning output
    classification_output = {
        "6 months (PRIMARY - clinical benefit)": classification_metrics.get('test', {})
    }
    with open(output_dir / 'classification_metrics.json', 'w') as f:
        json.dump(classification_output, f, indent=2)
    
    # Save hyperparameter search results
    with open(output_dir / 'cv_results.json', 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Save feature importance
    importance = model.get_score(importance_type='gain')
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    with open(output_dir / 'feature_importance.json', 'w') as f:
        json.dump(importance_sorted, f, indent=2)
    
    print(f"\n✅ Model saved to {output_dir}")


def main():
    print("=" * 70)
    print("XGBOOST TRAINING FOR FIRST TREATMENT STRATUM")
    print("Optimized for small datasets with CV and feature selection")
    print("=" * 70)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    
    # Feature selection (reduce from ~9500 to ~500)
    selected_features, prelim_model = feature_selection(X_train, y_train, n_features_target=500)
    
    # Hyperparameter tuning with 5-fold CV
    best_params, cv_results = hyperparameter_search_cv(
        X_train, y_train, selected_features, n_folds=5
    )
    
    # Train final model on full training set
    model, evals_result = train_final_model(
        X_train, y_train, X_val, y_val, selected_features, best_params
    )
    
    # Evaluate on all splits
    metrics, classification_metrics = evaluate_model(
        model, X_train, X_val, X_test, y_train, y_val, y_test, selected_features
    )
    
    # Save everything
    save_model(model, metrics, classification_metrics, selected_features, cv_results, best_params)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - Test AUROC: {:.4f}".format(
        classification_metrics.get('test', {}).get('auroc', 0.0)
    ))
    print("=" * 70)


if __name__ == '__main__':
    main()
