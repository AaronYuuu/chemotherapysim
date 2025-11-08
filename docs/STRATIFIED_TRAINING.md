# Stratified Training Guide

## Quick Start

```bash
# Train 2 models (recommended)
python train_stratified.py --strategy two_strata

# Or train 3 models  
python train_stratified.py --strategy three_strata

# Monitor progress
tail -f stratified_training.log
```

## What It Does

1. Loads full dataset
2. Splits by treatment line:
   - `two_strata`: Line 1 (822 samples) vs Line 2+ (2,112 samples)
   - `three_strata`: Line 1 (822) vs Line 2 (754) vs Line 3+ (1,358)
3. For each stratum: filters data, calls train.py, saves to `checkpoints_stratified/{stratum}/`

## Expected Results

After training completes (~2 hours), you'll have:
- `checkpoints_stratified/first_line/best_model.pt`
- `checkpoints_stratified/beyond_first_line/best_model.pt`

Each model learns genomic/drug effects **within** its treatment line context.

## Data Distribution

```
Treatment Line 1:  822 samples (28.0%) - Median PFS: 4.4 months, 38% ≥6mo
Treatment Line 2:  754 samples (25.7%) - Median PFS: 2.6 months, 29% ≥6mo
Treatment Line 3:  552 samples (18.8%) - Median PFS: 2.1 months, 19% ≥6mo
Treatment Line 4+: 806 samples (27.5%) - Progressively worse outcomes
```

All groups have **sufficient data** (>500 samples) for deep learning.

## Training Strategies

### Strategy 1: Two Strata (Recommended)

Train 2 models with excellent sample sizes:

```bash
python train_stratified.py --strategy two_strata
```

**Models:**
1. **First-line model** (822 samples)
   - Treatment-naive patients
   - Better baseline prognosis
   - Learn which drugs/genomics predict response in fresh patients

2. **Beyond-first-line model** (2,112 samples)
   - Pre-treated patients with potential resistance
   - Worse baseline prognosis  
   - Learn which drugs/genomics overcome resistance

### Strategy 2: Three Strata

Train 3 models for finer granularity:

```bash
python train_stratified.py --strategy three_strata
```

**Models:**
1. **First-line** (822 samples) - Treatment-naive
2. **Second-line** (754 samples) - One prior failure
3. **Third-line+** (1,358 samples) - Heavily pre-treated

## Expected Improvements

With stratified training, you should see:

✅ **Diverse predictions within each stratum** - Different drugs produce different PFS predictions for the same patient

✅ **Genomic features matter** - Mutations/CNAs influence which drugs are predicted to work

✅ **Drug-specific effects** - EGFR inhibitors work better with EGFR mutations, etc.

✅ **Better interpretability** - SHAP shows genomic/drug features, not just treatment_line

✅ **Clinically actionable** - Model can compare drug options for a specific patient at a specific line

## Output Structure

```
checkpoints_stratified/
├── first_line/
│   ├── best_model.pt              # First-line model weights
│   ├── config.json                # Configuration
│   └── training_curves.png        # Training history
├── beyond_first_line/
│   ├── best_model.pt              # Beyond-first-line weights
│   ├── config.json
│   └── training_curves.png
└── stratified_results.json        # Summary of all models
```

## Using Stratified Models in app.py

After training, update `app.py` to:
1. Load the appropriate model based on user's treatment_line input
2. Use first_line/best_model.pt for line=1
3. Use beyond_first_line/best_model.pt for line≥2

Example modification:
```python
# In app.py, load_model() function:
treatment_line = st.session_state.get('treatment_line', 1)

if treatment_line == 1:
    checkpoint_path = "checkpoints_stratified/first_line/best_model.pt"
else:
    checkpoint_path = "checkpoints_stratified/beyond_first_line/best_model.pt"

checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Training Time

Expect similar training time to original:
- **First-line**: ~30-40 minutes (smaller dataset)
- **Beyond-first-line**: ~60-80 minutes (larger dataset)
- **Total**: ~2 hours for both models

## Validation

Compare stratified vs original model:

**Original model:**
- Test AUROC: 0.7436
- Predictions dominated by treatment_line
- Narrow range within each line (e.g., 1.5-1.8 for line 1)

**Stratified models (expected):**
- Similar or better AUROC within each stratum
- **Wider prediction range** within each line (e.g., 0.5-8.0 for line 1)
- Genomics and drugs drive predictions, not just treatment line
- Better clinical utility for drug selection

## Next Steps

1. **Run stratified training:**
   ```bash
   python train_stratified.py --strategy two_strata
   ```

2. **Compare results** in `checkpoints_stratified/stratified_results.json`

3. **Update app.py** to use stratified models

4. **Test in Streamlit:** Try different drugs with same patient - should see different predictions now!

5. **Analyze SHAP:** Feature importance should now show genomic/drug features, not just treatment_line
