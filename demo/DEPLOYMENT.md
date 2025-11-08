# Deploying to Streamlit Cloud

## Prerequisites

Before deploying, ensure:

1. **Data files are committed:**
   - `data/processed/drug_smiles_cache.json`
   - `data/processed/drug_classes.json`

2. **Artifacts are committed:**
   - `artifacts/feature_names.json`
   - `artifacts/sample_metadata.csv`
   - `artifacts/drug_fp_library.npz`
   - `artifacts/drug_map.json`

3. **Trained models are uploaded:**
   - `checkpoints_stratified/previous_treatment/` (Deep Learning model)
   - `checkpoints_stratified/first_treatment_xgboost/` (XGBoost model)

## Deployment Steps

### 1. Commit Required Files

```bash
# Allow necessary data files through gitignore
git add data/processed/drug_smiles_cache.json
git add data/processed/drug_classes.json

# Commit artifacts (you may need to use Git LFS for large files)
git add artifacts/feature_names.json
git add artifacts/sample_metadata.csv
git add artifacts/drug_fp_library.npz
git add artifacts/drug_map.json

git commit -m "Add data and artifacts for deployment"
```

### 2. Handle Large Model Files

Model checkpoints are too large for regular git. Options:

**Option A: Git LFS (Recommended)**
```bash
git lfs install
git lfs track "checkpoints_stratified/**/*.pt"
git lfs track "checkpoints_stratified/**/*.pkl"
git add .gitattributes
git add checkpoints_stratified/
git commit -m "Add model checkpoints with Git LFS"
```

**Option B: External Storage**
- Upload models to cloud storage (AWS S3, Google Cloud Storage, etc.)
- Modify `load_stratified_models()` in `app.py` to download from URLs

**Option C: Streamlit Secrets (For Small Files)**
- Upload models as base64-encoded secrets
- Decode in app at runtime

### 3. Configure Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set main file path: `demo/app.py`
4. Set Python version: 3.10 or 3.11
5. Deploy!

### 4. Add Secrets (Optional)

If using external storage, add to Streamlit secrets:

```toml
# .streamlit/secrets.toml (add in Streamlit Cloud settings)
[models]
dl_model_url = "https://your-storage.com/previous_treatment/best_model.pt"
xgb_model_url = "https://your-storage.com/first_treatment_xgboost/xgboost_model.pkl"
```

## File Size Limits

- GitHub: 100 MB per file (without LFS)
- Git LFS: 1 GB per file
- Streamlit Cloud: 1 GB total app size

## Troubleshooting

### "No module named 'src'"
- Ensure `sys.path.insert(0, str(PROJECT_ROOT))` is in app.py
- Check that `src/models/__init__.py` exists

### "No such file or directory"
- Verify files are committed: `git ls-files | grep data/processed`
- Check paths use `PROJECT_ROOT` correctly

### "XGBoost not installed"
- Ensure `xgboost>=2.0.0` is in `demo/requirements.txt`
- Streamlit Cloud will reinstall on next deployment

### "Model not loading"
- Check model files are < 100 MB (use Git LFS if larger)
- Verify checkpoint paths in `CHECKPOINT_DIR_*` variables

## Quick Deploy (Minimal Setup)

For demo purposes without models:

1. Comment out model loading in `load_stratified_models()`
2. Use mock predictions for demonstration
3. Display warning: "Demo mode - using simulated predictions"

```python
@st.cache_resource
def load_stratified_models():
    st.warning("⚠️ Running in DEMO MODE - predictions are simulated")
    return None, None, {}, {}
```
