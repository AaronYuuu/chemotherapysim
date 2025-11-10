import sys
sys.path.insert(0, 'src')
import pickle
import numpy as np
import json

# Load feature names
with open('artifacts/feature_names.json', 'r') as f:
    feature_data = json.load(f)
    all_feature_names = feature_data.get('tabular_features', [])

# Load feature selection metadata to get selected indices
with open('artifacts/feature_selection_metadata.pkl', 'rb') as f:
    feat_meta = pickle.load(f)
    selected_indices = feat_meta['selected_indices']

# Map selected indices to feature names
selected_feature_names = [all_feature_names[i] for i in selected_indices if i < len(all_feature_names)]

print(f'Loaded {len(all_feature_names)} total feature names')
print(f'Selected {len(selected_feature_names)} features')

# Load ensemble
with open('artifacts/models/final_optimized_ensemble.pkl', 'rb') as f:
    ensemble = pickle.load(f)

models = ensemble['models']
xgb_model = models['xgboost']
ridge_model = models['ridge']
weights = ensemble['weights']

# Get XGBoost feature importance
xgb_booster = xgb_model.get_booster()
xgb_importance = xgb_booster.get_score(importance_type='gain')

# Get Ridge coefficients
ridge_coefs = ridge_model.coef_

# Sort by importance
xgb_sorted = sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True)
ridge_sorted_indices = np.argsort(np.abs(ridge_coefs))[::-1]

with open('feature_importance_report.txt', 'w') as f:
    f.write('FEATURE IMPORTANCE REPORT\n')
    f.write('='*80 + '\n\n')
    
    f.write('ENSEMBLE COMPOSITION:\n')
    f.write('  - XGBoost:   46.5% (tree-based, interpretable)\n')
    f.write('  - Attention: 31.3% (neural network)\n')
    f.write('  - Ridge:     22.2% (linear model)\n\n')
    f.write('Total Features: 150 (selected from 244 engineered features)\n\n')

    f.write('='*80 + '\n')
    f.write('TOP 30 FEATURES BY XGBOOST (46.5% of prediction)\n')
    f.write('='*80 + '\n')
    f.write('XGBoost Gain = average improvement when splitting on this feature\n\n')
    f.write(f'{"Rank":>4} {"Feature":>50} {"Gain":>12} {"Importance":>12}\n')
    f.write('-'*80 + '\n')

    total_gain = sum([s for _, s in xgb_sorted])
    for rank, (feat_key, score) in enumerate(xgb_sorted[:30], 1):
        feat_idx = int(feat_key.replace('f', ''))
        feat_name = selected_feature_names[feat_idx] if feat_idx < len(selected_feature_names) else f'Feature_{feat_idx}'
        pct = (score / total_gain) * 100
        f.write(f'{rank:4d} {feat_name:50s} {score:12.2f}    {pct:10.2f}%\n')

    f.write('\n' + '='*80 + '\n')
    f.write('TOP 30 FEATURES BY RIDGE COEFFICIENTS (22.2% of prediction)\n')
    f.write('='*80 + '\n')
    f.write('Positive coef = increases PFS | Negative coef = decreases PFS\n\n')
    f.write(f'{"Rank":>4} {"Feature":>50} {"Coefficient":>14} {"Effect":>15}\n')
    f.write('-'*80 + '\n')

    for rank, idx in enumerate(ridge_sorted_indices[:30], 1):
        feat_name = selected_feature_names[idx] if idx < len(selected_feature_names) else f'Feature_{idx}'
        coef = ridge_coefs[idx]
        effect = 'Increases PFS' if coef > 0 else 'Decreases PFS'
        f.write(f'{rank:4d} {feat_name:50s} {coef:+14.6f}   {effect:>15}\n')

    f.write('\n' + '='*80 + '\n')
    f.write('ATTENTION MODEL (31.3% of prediction)\n')
    f.write('='*80 + '\n')
    f.write('Architecture: 150 -> 256 -> 256 -> 256 -> 1\n')
    f.write('Attention Heads: 8\n')
    f.write('Parameters: ~1.9M\n\n')
    f.write('Uses all 150 features with:\n')
    f.write('  - Multi-head self-attention (8 heads learn different patterns)\n')
    f.write('  - Non-linear transformations\n')
    f.write('  - Sample-specific attention weights (varies per patient)\n\n')
    f.write('Cannot show global importance (attention is patient-specific)\n')

    f.write('\n' + '='*80 + '\n')
    f.write('FEATURE SELECTION SUMMARY\n')
    f.write('='*80 + '\n')
    f.write('Original: 244 features\n')
    f.write('Selected: 150 features (38.5% reduction)\n\n')
    f.write('Selection methods (ensemble voting):\n')
    f.write('  1. Mutual Information (non-linear relevance)\n')
    f.write('  2. Recursive Feature Elimination (RFE)\n')
    f.write('  3. Stability Selection (bootstrap consistency)\n')
    f.write('  4. Permutation Importance\n\n')
    f.write('Features kept if selected by >= 3 of 4 methods\n')

    f.write('\n' + '='*80 + '\n')
    f.write('HOW TO INTERPRET\n')
    f.write('='*80 + '\n\n')
    f.write('For any prediction:\n')
    f.write('  Final = 0.465*XGBoost + 0.222*Ridge + 0.313*Attention\n\n')
    f.write('XGBoost contribution:\n')
    f.write('  - Follows tree rules based on top features above\n')
    f.write('  - Can trace exact decision path for any prediction\n\n')
    f.write('Ridge contribution:\n')
    f.write('  - Linear sum: intercept + sum(feature * coefficient)\n')
    f.write('  - Positive coefs push PFS up, negative push down\n\n')
    f.write('Attention contribution:\n')
    f.write('  - Complex non-linear combination\n')
    f.write('  - Different features important for different patients\n')

print('Report saved to: feature_importance_report.txt')
