# Project Structure

## Directories

### `/src` - Source Code
Core application code organized by functionality:

- **`/src/models`** - Model architectures
  - `deep_learning.py` - Deep learning model (23M parameters) for previous treatment
  - `utils.py` - Utility functions for data processing
  
- **`/src/training`** - Training scripts
  - `train_deep_learning.py` - Train deep learning model (line ≥ 2)
  - `train_xgboost.py` - Train XGBoost model (line = 1)

### `/demo` - Web Application
Streamlit web interface for interactive predictions:
- `app.py` - Main Streamlit application
- `requirements.txt` - Demo-specific dependencies

**Run the demo:**
```bash
cd demo
pip install -r requirements.txt
streamlit run app.py
```

### `/notebooks` - Jupyter Notebooks
Exploratory data analysis and feature engineering:
- `dataprocess.ipynb` - Data preprocessing pipeline
- `feature_engineering.ipynb` - Feature extraction and selection

### `/docs` - Documentation
Project documentation and guides:
- `TRAINING_GUIDE.md` - Comprehensive training instructions
- `METRICS_RATIONALE.md` - Explanation of evaluation metrics
- `PREDICTION_RANGE_EXPLANATION.md` - Understanding prediction outputs
- `STRATIFIED_TRAINING.md` - Stratified modeling approach

### `/data` - Data Files
- `/processed` - Preprocessed datasets
- Drug SMILES, feature names, sample metadata

### `/artifacts` - Model Artifacts
Pre-computed resources:
- Drug fingerprint libraries
- Feature names and statistics
- Sample metadata

### `/checkpoints_stratified` - Trained Models
Saved model checkpoints:
- `/first_treatment_xgboost` - XGBoost model for line 1
- `/previous_treatment` - Deep learning model for line ≥ 2

### `/logs` - Training Logs
Training outputs and metrics

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
# Train XGBoost (first treatment)
python src/training/train_xgboost.py

# Train Deep Learning (previous treatment)
python src/training/train_deep_learning.py
```

### 3. Run Demo
```bash
streamlit run demo/app.py
```

## File Organization Benefits

✅ **Separation of Concerns:** Training, models, and demo are isolated  
✅ **Cleaner Imports:** Organized module structure  
✅ **Documentation Hub:** All docs in `/docs`  
✅ **Easy Navigation:** Intuitive directory names  
✅ **Scalability:** Easy to add new models or features
