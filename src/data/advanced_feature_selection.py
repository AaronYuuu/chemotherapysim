"""
Advanced Feature Selection for PFS Prediction

Uses multiple techniques:
1. Mutual Information for non-linear relevance
2. Recursive Feature Elimination (RFE)
3. Stability selection
4. Permutation importance
"""

import numpy as np
from sklearn.feature_selection import (
    mutual_info_regression,
    f_regression
)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
import pickle


def select_by_mutual_information(X, y, k=150):
    """
    Select features by mutual information (captures non-linear relationships).
    """
    print(f"\nMutual Information Feature Selection (top {k})...")
    
    mi_scores = mutual_info_regression(X, y, random_state=42)
    
    # Select top k
    top_indices = np.argsort(mi_scores)[-k:]
    
    print(f"  Selected {k} features")
    print(f"  MI range: [{mi_scores[top_indices].min():.4f}, {mi_scores[top_indices].max():.4f}]")
    
    return top_indices, mi_scores


def select_by_rfe(X, y, k=150):
    """
    Recursive Feature Elimination with Gradient Boosting.
    """
    print(f"\nRecursive Feature Elimination (top {k})...")
    
    from sklearn.feature_selection import RFE
    
    estimator = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        random_state=42
    )
    
    selector = RFE(estimator, n_features_to_select=k, step=10)
    selector.fit(X, y)
    
    selected_indices = np.where(selector.support_)[0]
    
    print(f"  Selected {len(selected_indices)} features")
    
    return selected_indices, selector.ranking_


def select_by_stability(X, y, k=150, n_bootstraps=50):
    """
    Stability selection: features selected consistently across bootstrap samples.
    """
    print(f"\nStability Selection ({n_bootstraps} bootstraps, top {k})...")
    
    n_samples, n_features = X.shape
    selection_counts = np.zeros(n_features)
    
    for i in range(n_bootstraps):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Select features using mutual information
        mi_scores = mutual_info_regression(X_boot, y_boot, random_state=i)
        top_k = np.argsort(mi_scores)[-k:]
        selection_counts[top_k] += 1
        
        if (i + 1) % 10 == 0:
            print(f"    Bootstrap {i+1}/{n_bootstraps}")
    
    # Select features with high selection frequency
    selection_freq = selection_counts / n_bootstraps
    stable_indices = np.where(selection_freq > 0.5)[0]  # Selected in >50% of bootstraps
    
    print(f"  {len(stable_indices)} features selected in >50% of bootstraps")
    
    # If too few, take top k by frequency
    if len(stable_indices) < k:
        stable_indices = np.argsort(selection_freq)[-k:]
        print(f"  Expanded to {k} features")
    
    return stable_indices, selection_freq


def select_by_permutation_importance(X, y, k=150):
    """
    Permutation importance: measure feature importance by shuffling.
    """
    print(f"\nPermutation Importance (top {k})...")
    
    # Train a model
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X, y)
    
    # Calculate permutation importance
    result = permutation_importance(
        model, X, y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    importances = result.importances_mean
    top_indices = np.argsort(importances)[-k:]
    
    print(f"  Selected {k} features")
    print(f"  Importance range: [{importances[top_indices].min():.4f}, {importances[top_indices].max():.4f}]")
    
    return top_indices, importances


def ensemble_feature_selection(X, y, k=150):
    """
    Combine multiple feature selection methods by voting.
    """
    print("=" * 60)
    print("Ensemble Feature Selection")
    print("=" * 60)
    
    n_features = X.shape[1]
    
    # Run all methods
    mi_indices, mi_scores = select_by_mutual_information(X, y, k=k)
    rfe_indices, rfe_ranking = select_by_rfe(X, y, k=k)
    stable_indices, stability_freq = select_by_stability(X, y, k=k, n_bootstraps=30)
    perm_indices, perm_importance = select_by_permutation_importance(X, y, k=k)
    
    # Vote-based selection
    print("\nEnsemble voting...")
    votes = np.zeros(n_features)
    votes[mi_indices] += 1
    votes[rfe_indices] += 1
    votes[stable_indices] += 1
    votes[perm_indices] += 1
    
    # Features with >=3 votes (selected by at least 3 methods)
    high_confidence = np.where(votes >= 3)[0]
    
    # Features with >=2 votes
    medium_confidence = np.where(votes >= 2)[0]
    
    print(f"  High confidence (>=3 votes): {len(high_confidence)} features")
    print(f"  Medium confidence (>=2 votes): {len(medium_confidence)} features")
    
    # Select final features
    if len(high_confidence) >= k:
        selected_indices = high_confidence[:k]
    elif len(medium_confidence) >= k:
        # Take all high confidence + top medium confidence
        remaining = k - len(high_confidence)
        medium_only = np.setdiff1d(medium_confidence, high_confidence)
        selected_indices = np.concatenate([high_confidence, medium_only[:remaining]])
    else:
        # Take top k by total votes
        selected_indices = np.argsort(votes)[-k:]
    
    print(f"\nFinal selection: {len(selected_indices)} features")
    
    # Create comprehensive scores
    scores = {
        'mutual_information': mi_scores,
        'rfe_ranking': rfe_ranking,
        'stability_frequency': stability_freq,
        'permutation_importance': perm_importance,
        'ensemble_votes': votes
    }
    
    return selected_indices, scores


def main():
    """
    Main feature selection pipeline.
    """
    # Load data
    print("Loading engineered features...")
    X = np.load('artifacts/X_engineered.npy')
    y = np.load('artifacts/y_engineered.npy')
    
    print(f"  Original: {X.shape}")
    
    # Load metadata for feature names
    try:
        with open('artifacts/feature_engineering_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        feature_names = metadata.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])
    except:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Select features
    k = 150  # Target number of features
    selected_indices, scores = ensemble_feature_selection(X, y, k=k)
    
    # Create reduced dataset
    X_selected = X[:, selected_indices]
    
    print("\n" + "=" * 60)
    print("Selected Features Summary")
    print("=" * 60)
    
    selected_names = [feature_names[i] for i in selected_indices]
    
    # Top 20 features by ensemble votes
    vote_order = np.argsort(scores['ensemble_votes'][selected_indices])[::-1]
    print("\nTop 20 features by ensemble votes:")
    for rank, idx in enumerate(vote_order[:20], 1):
        feat_idx = selected_indices[idx]
        name = feature_names[feat_idx]
        votes = scores['ensemble_votes'][feat_idx]
        mi = scores['mutual_information'][feat_idx]
        perm = scores['permutation_importance'][feat_idx]
        print(f"  {rank:2d}. {name:40s} (votes={votes:.0f}, MI={mi:.4f}, perm={perm:.4f})")
    
    # Save
    print("\nSaving selected features...")
    np.save('artifacts/X_selected.npy', X_selected)
    
    selection_metadata = {
        'selected_indices': selected_indices,
        'selected_names': selected_names,
        'scores': scores,
        'k': k
    }
    
    with open('artifacts/feature_selection_metadata.pkl', 'wb') as f:
        pickle.dump(selection_metadata, f)
    
    print(f"  Saved: artifacts/X_selected.npy ({X_selected.shape})")
    print(f"  Metadata: artifacts/feature_selection_metadata.pkl")
    
    print("\n" + "=" * 60)
    print("Feature selection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
