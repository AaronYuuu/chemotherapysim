"""
Precision Oncology Chemotherapy Sensitivity Predictor
======================================================
Streamlit web app for predicting PFS and treatment resistance using
patient omics and clinical features with stratified models:
  - First treatment (line=1): XGBoost (optimized for N=822)
  - Previous treatment (line>=2): Deep Learning (23M parameters)

Requirements:
    streamlit>=1.28.0
    torch>=2.0.0
    pandas>=2.0.0
    numpy>=1.24.0
    plotly>=5.17.0
    rdkit>=2023.0.0
    scikit-learn>=1.3.0
    xgboost>=2.0.0
    shap>=0.43.0 (optional)

Run:
    streamlit run demo/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for model import
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import models
try:
    from models.deep_learning import ImprovedDrugResponseModel
    HAS_DL_MODEL = True
except ImportError as e:
    st.error(f"❌ Could not import Deep Learning model: {e}")
    HAS_DL_MODEL = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    st.warning("⚠️ XGBoost not installed. First treatment predictions will be disabled.")

# Try to import optional dependencies
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    st.warning("⚠️ RDKit not installed. Custom SMILES input will be disabled.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ===================================================================
# CONFIGURATION
# ===================================================================

# Project paths (already defined above as PROJECT_ROOT)
# Model paths - stratified by treatment line
CHECKPOINT_DIR_PREVIOUS = PROJECT_ROOT / "checkpoints_stratified" / "previous_treatment"
CHECKPOINT_DIR_FIRST_XGB = PROJECT_ROOT / "checkpoints_stratified" / "first_treatment_xgboost"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Page config
st.set_page_config(
    page_title="Precision Oncology Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===================================================================
# MODEL LOADING FUNCTIONS
# ===================================================================

@st.cache_resource
def load_stratified_models() -> Tuple[Optional[nn.Module], Optional[object], Dict, Dict]:
    """
    Load both stratified models:
      - Deep Learning for previous treatment (line >= 2)
      - XGBoost for first treatment (line == 1)
    
    Returns:
        dl_model: PyTorch deep learning model (or None if unavailable)
        xgb_model: XGBoost model (or None if unavailable)
        dl_config: Deep learning configuration
        xgb_config: XGBoost configuration
    """
    dl_model = None
    xgb_model = None
    dl_config = {}
    xgb_config = {}
    
    # Load Deep Learning model for previous treatment
    if HAS_DL_MODEL and CHECKPOINT_DIR_PREVIOUS.exists():
        try:
            config_path = CHECKPOINT_DIR_PREVIOUS / "config.json"
            with open(config_path, 'r') as f:
                dl_config = json.load(f)
            
            dl_model = ImprovedDrugResponseModel(
                genomic_dim=dl_config.get("genomic_dim", 1318),
                drug_fp_dim=dl_config.get("drug_fp_dim", 8192),
                embed_dim=dl_config.get("embed_dim", 256),
                dropout_genomic=dl_config.get("dropout_genomic", 0.5),
                dropout_drug=dl_config.get("dropout_drug", 0.4),
                dropout_head=dl_config.get("dropout_head", 0.5)
            )
            
            checkpoint_path = CHECKPOINT_DIR_PREVIOUS / "best_model.pt"
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            dl_model.load_state_dict(checkpoint['model_state_dict'])
            dl_model.eval()
            
            st.success("✅ Deep Learning model loaded (previous treatment)")
        except Exception as e:
            st.warning(f"⚠️ Could not load Deep Learning model: {str(e)}")
    elif not HAS_DL_MODEL:
        st.warning(f"⚠️ Deep Learning model import failed")
    elif not CHECKPOINT_DIR_PREVIOUS.exists():
        st.warning(f"⚠️ Deep Learning checkpoint not found at: {CHECKPOINT_DIR_PREVIOUS}")
    
    # Load XGBoost model for first treatment
    if HAS_XGBOOST and CHECKPOINT_DIR_FIRST_XGB.exists():
        try:
            model_path = CHECKPOINT_DIR_FIRST_XGB / "xgboost_model.pkl"
            import pickle
            with open(model_path, 'rb') as f:
                xgb_model = pickle.load(f)
            
            # Load selected features
            features_path = CHECKPOINT_DIR_FIRST_XGB / "selected_features.npy"
            xgb_config['selected_features'] = np.load(features_path)
            
            # Load best params
            params_path = CHECKPOINT_DIR_FIRST_XGB / "best_params.json"
            with open(params_path, 'r') as f:
                xgb_config['params'] = json.load(f)
            
            st.success("✅ XGBoost model loaded (first treatment)")
        except Exception as e:
            st.warning(f"⚠️ Could not load XGBoost model: {str(e)}")
    elif not HAS_XGBOOST:
        st.warning(f"⚠️ XGBoost not installed")
    elif not CHECKPOINT_DIR_FIRST_XGB.exists():
        st.warning(f"⚠️ XGBoost checkpoint not found at: {CHECKPOINT_DIR_FIRST_XGB}")
    
    return dl_model, xgb_model, dl_config, xgb_config


@st.cache_data
def load_drug_library() -> Tuple[Dict, Dict, np.ndarray]:
    """Load drug SMILES codes, fingerprint library, and drug classes."""
    try:
        # Load SMILES codes
        smiles_path = DATA_DIR / "drug_smiles_cache.json"
        with open(smiles_path, 'r') as f:
            drug_smiles = json.load(f)
        
        # Load drug classes
        classes_path = DATA_DIR / "drug_classes.json"
        with open(classes_path, 'r') as f:
            drug_classes = json.load(f)
        
        # Load pre-computed fingerprint library (optional)
        fp_library = None
        drug_map = {}
        
        fp_library_path = ARTIFACTS_DIR / "drug_fp_library.npz"
        if fp_library_path.exists():
            try:
                fp_data = np.load(fp_library_path, allow_pickle=True)
                # Check what keys are available
                available_keys = list(fp_data.keys())
                
                # Try common key names
                if 'fingerprints' in available_keys:
                    fp_library = fp_data['fingerprints']
                elif 'arr_0' in available_keys:
                    fp_library = fp_data['arr_0']
                elif len(available_keys) > 0:
                    # Use the first available array
                    fp_library = fp_data[available_keys[0]]
                
                # Load drug mapping
                drug_map_path = ARTIFACTS_DIR / "drug_map.json"
                if drug_map_path.exists():
                    with open(drug_map_path, 'r') as f:
                        drug_map = json.load(f)
            except Exception as e:
                st.warning(f"⚠️ Could not load pre-computed fingerprints: {str(e)}")
                fp_library = None
                drug_map = {}
        
        return drug_smiles, drug_classes, fp_library, drug_map
    except Exception as e:
        st.error(f"❌ Error loading drug library: {str(e)}")
        return {}, {}, None, {}


@st.cache_data
def load_feature_info() -> Tuple[List[str], pd.DataFrame]:
    """Load feature names and sample statistics."""
    try:
        # Load feature names
        feature_path = ARTIFACTS_DIR / "feature_names.json"
        with open(feature_path, 'r') as f:
            feature_data = json.load(f)
        feature_names = feature_data.get('tabular_features', [])
        
        # Load sample metadata for statistics
        metadata_path = ARTIFACTS_DIR / "sample_metadata.csv"
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
        else:
            metadata = pd.DataFrame()
        
        return feature_names, metadata
    except Exception as e:
        st.warning(f"⚠️ Could not load feature info: {str(e)}")
        return [], pd.DataFrame()



@st.cache_data
def load_classification_metrics(treatment_line: int = 1) -> Dict:
    """Load saved classification metrics from test set for appropriate model."""
    try:
        if treatment_line == 1:
            metrics_path = CHECKPOINT_DIR_FIRST_XGB / "classification_metrics.json"
        else:
            metrics_path = CHECKPOINT_DIR_PREVIOUS / "classification_metrics.json"
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        st.warning(f"⚠️ Could not load metrics: {str(e)}")
        return {}


# ===================================================================
# PREDICTION FUNCTIONS - STRATIFIED BY TREATMENT LINE
# ===================================================================

def predict_pfs_stratified(
    dl_model: Optional[nn.Module],
    xgb_model: Optional[object],
    genomic_features: np.ndarray,
    drug_fingerprint: np.ndarray,
    treatment_line: int,
    xgb_config: Dict
) -> Tuple[float, float, float, float, str]:
    """
    Make prediction using appropriate model based on treatment line.
    
    Args:
        dl_model: Deep learning model (for line >= 2)
        xgb_model: XGBoost model (for line == 1)
        genomic_features: Genomic feature vector
        drug_fingerprint: Drug fingerprint vector
        treatment_line: Treatment line number
        xgb_config: XGBoost configuration with selected features
    
    Returns:
        pfs_months: Predicted PFS in months
        resistance_prob: Resistance probability at 6 months (%)
        log_pred: Raw log-space prediction
        confidence: Confidence score (0-1)
        model_used: Which model was used ("XGBoost" or "Deep Learning")
    """
    
    # First treatment: Use XGBoost
    if treatment_line == 1:
        if xgb_model is None:
            raise ValueError("XGBoost model not available for first treatment prediction")
        
        # Combine features for XGBoost
        X_combined = np.concatenate([genomic_features, drug_fingerprint])
        
        # Select features used by XGBoost
        selected_features = xgb_config.get('selected_features', np.arange(len(X_combined)))
        X_selected = X_combined[selected_features].reshape(1, -1)
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(X_selected)
        
        # Predict (in log-space)
        log_pred = xgb_model.predict(dmatrix)[0]
        
        # Convert to months
        pfs_months = np.exp(log_pred) - 1
        
        # Calculate resistance probability
        threshold_log = np.log1p(6.0)
        prob_success = 1 / (1 + np.exp(-2 * (log_pred - threshold_log)))
        resistance_prob = (1 - prob_success) * 100
        
        # Calculate confidence - improved for XGBoost
        # Predictions further from threshold (1.946) get higher confidence
        distance_from_threshold = abs(log_pred - threshold_log)
        # Scale: 0.5 units away = 0.6 confidence, 1.0 units = 0.8, 2.0 units = 0.95
        confidence = 1.0 - np.exp(-0.8 * distance_from_threshold)
        confidence = max(0.3, min(confidence, 0.99))  # Floor at 0.3, cap at 0.99
        
        model_used = "XGBoost (First Treatment)"
    
    # Previous treatment: Use Deep Learning
    else:
        if dl_model is None:
            raise ValueError("Deep Learning model not available for previous treatment prediction")
        
        with torch.no_grad():
            genomic_tensor = torch.FloatTensor(genomic_features).unsqueeze(0)
            drug_tensor = torch.FloatTensor(drug_fingerprint).unsqueeze(0)
            
            log_pred = dl_model(genomic_tensor, drug_tensor).item()
            
            pfs_months = np.exp(log_pred) - 1
            
            threshold_log = np.log1p(6.0)
            prob_success = 1 / (1 + np.exp(-2 * (log_pred - threshold_log)))
            resistance_prob = (1 - prob_success) * 100
            
            # Calculate confidence - improved for Deep Learning
            distance_from_threshold = abs(log_pred - threshold_log)
            confidence = 1.0 - np.exp(-1.5 * distance_from_threshold)
            confidence = max(0.3, min(confidence, 0.99))  # Floor at 0.3, cap at 0.99
        
        model_used = "Deep Learning (Previous Treatment)"
    
    return pfs_months, resistance_prob, log_pred, confidence, model_used

# ===================================================================
# DRUG FINGERPRINT GENERATION
# ===================================================================

def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint."""
    if not HAS_RDKIT:
        st.error("RDKit is required for custom SMILES input")
        return None
    
    if smiles is None or smiles.strip() == "":
        return np.zeros(n_bits, dtype=np.float32)
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.warning(f"⚠️ Invalid SMILES: {smiles}")
            return np.zeros(n_bits, dtype=np.float32)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        for idx in fp.GetOnBits():
            arr[idx] = 1.0
        return arr
    except Exception as e:
        st.warning(f"⚠️ Error generating fingerprint: {str(e)}")
        return np.zeros(n_bits, dtype=np.float32)


def create_drug_fingerprint_vector(
    drugs: List[str],
    drug_smiles: Dict[str, str],
    custom_smiles: Dict[str, str],
    fp_library: Optional[np.ndarray] = None,
    drug_map: Optional[Dict] = None
) -> np.ndarray:
    """
    Create 8192-dim drug fingerprint vector for up to 4 drugs.
    Uses pre-computed library when available, otherwise generates on-the-fly.
    """
    n_positions = 4
    fp_dim = 2048
    total_dim = n_positions * fp_dim
    
    fingerprint_vector = np.zeros(total_dim, dtype=np.float32)
    
    for i, drug_name in enumerate(drugs[:n_positions]):
        start_idx = i * fp_dim
        end_idx = (i + 1) * fp_dim
        
        # Check if custom SMILES provided
        if drug_name in custom_smiles and custom_smiles[drug_name]:
            fp = smiles_to_fingerprint(custom_smiles[drug_name], n_bits=fp_dim)
            if fp is not None:
                fingerprint_vector[start_idx:end_idx] = fp
        # Check pre-computed library
        elif fp_library is not None and drug_map is not None and drug_name in drug_map:
            try:
                drug_idx = drug_map[drug_name]
                # Convert to int if it's a string
                if isinstance(drug_idx, str):
                    drug_idx = int(drug_idx)
                fingerprint_vector[start_idx:end_idx] = fp_library[drug_idx]
            except (ValueError, IndexError, TypeError) as e:
                # If pre-computed lookup fails, fall back to generating from SMILES
                if drug_name in drug_smiles:
                    smiles = drug_smiles[drug_name]
                    fp = smiles_to_fingerprint(smiles, n_bits=fp_dim)
                    if fp is not None:
                        fingerprint_vector[start_idx:end_idx] = fp
        # Generate from SMILES cache
        elif drug_name in drug_smiles:
            smiles = drug_smiles[drug_name]
            fp = smiles_to_fingerprint(smiles, n_bits=fp_dim)
            if fp is not None:
                fingerprint_vector[start_idx:end_idx] = fp
        else:
            # Drug not found, leave as zeros
            pass
    
    return fingerprint_vector


# ===================================================================
# FEATURE VECTOR CREATION
# ===================================================================

def create_genomic_feature_vector(
    feature_names: List[str],
    user_inputs: Dict,
    metadata: pd.DataFrame
) -> np.ndarray:
    """
    Create genomic + clinical feature vector from user inputs.
    Uses median values from training data as defaults.
    """
    feature_vector = np.zeros(len(feature_names), dtype=np.float32)
    
    # Calculate medians from metadata if available
    if not metadata.empty:
        medians = {}
        for col in metadata.columns:
            if pd.api.types.is_numeric_dtype(metadata[col]):
                medians[col] = metadata[col].median()
    else:
        medians = {}
    
    for i, feature_name in enumerate(feature_names):
        # Check if user provided this feature
        if feature_name in user_inputs:
            feature_vector[i] = user_inputs[feature_name]
        # Use median as default
        elif feature_name in medians:
            feature_vector[i] = medians[feature_name]
        # Default to 0
        else:
            feature_vector[i] = 0.0
    
    return feature_vector


# ===================================================================
# PREDICTION & INTERPRETATION
# ===================================================================



def get_feature_importance_shap(
    model: nn.Module,
    genomic_features: np.ndarray,
    drug_fingerprint: np.ndarray,
    feature_names: List[str],
    top_k: int = 15,
    is_xgboost: bool = False,
    xgb_config: Dict = None
) -> Tuple[pd.DataFrame, Optional[object]]:
    """
    Calculate feature importance using SHAP values.
    Works with both Deep Learning and XGBoost models.
    
    Returns:
        importance_df: DataFrame with feature names, SHAP values, and direction
        shap_explanation: SHAP explanation object (for advanced plotting)
    """
    if not HAS_SHAP:
        return get_feature_importance_mock(feature_names, genomic_features, top_k), None
    
    def parse_shap_value(val):
        """Parse SHAP value that might be in string scientific notation format."""
        try:
            # If it's already a number, convert directly
            return float(val)
        except (ValueError, TypeError):
            # Handle string scientific notation like '[1.673709E0]'
            try:
                # Remove brackets if present
                val_str = str(val).strip('[]')
                
                # Check if it contains 'E' or 'e' for scientific notation
                if 'E' in val_str.upper():
                    # Split by 'E' or 'e'
                    parts = val_str.upper().split('E')
                    if len(parts) == 2:
                        mantissa = float(parts[0])
                        exponent = int(parts[1])
                        return mantissa * (10 ** exponent)
                
                # If no 'E', just try to parse as float
                return float(val_str)
            except Exception:
                # Last resort: return 0
                return 0.0
    
    try:
        if is_xgboost:
            # For XGBoost, use TreeExplainer (faster and more accurate)
            explainer = shap.TreeExplainer(model)
            
            # Combine genomic and drug features
            X_combined = np.concatenate([genomic_features, drug_fingerprint])
            
            # Select features used by XGBoost
            if xgb_config is not None and 'selected_features' in xgb_config:
                selected_features = xgb_config['selected_features']
                X_selected = X_combined[selected_features].reshape(1, -1)
            else:
                X_selected = X_combined.reshape(1, -1)
            
            # Calculate SHAP values - ensure float64
            X_selected = X_selected.astype(np.float64)
            shap_values = explainer.shap_values(X_selected)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
            
            # Parse each value individually to handle string formats
            shap_values_parsed = []
            for val in shap_values:
                parsed_val = parse_shap_value(val)
                shap_values_parsed.append(parsed_val)
            
            shap_values = np.array(shap_values_parsed, dtype=np.float64)
            
            # Match with feature names (only for selected features)
            importances = []
            for idx, shap_val in enumerate(shap_values):
                if abs(shap_val) > 1e-6:
                    # Get original feature index
                    if xgb_config is not None and 'selected_features' in xgb_config:
                        orig_idx = selected_features[idx]
                        if orig_idx < len(feature_names):
                            feature_name = feature_names[orig_idx]
                            feature_value = X_combined[orig_idx]
                        else:
                            feature_name = f"Feature_{orig_idx}"
                            feature_value = X_combined[orig_idx] if orig_idx < len(X_combined) else 0.0
                    else:
                        feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                        feature_value = X_selected[0, idx]
                    
                    importances.append({
                        'Feature': feature_name,
                        'SHAP Value': float(shap_val),
                        'Impact': abs(float(shap_val)),
                        'Direction': 'Increases PFS' if shap_val > 0 else 'Decreases PFS',
                        'Feature Value': float(feature_value)
                    })
        else:
            # Deep Learning model - use KernelExplainer
            def model_predict(genomic_batch):
                """Wrapper that fixes drug fingerprint and varies genomic features."""
                genomic_tensor = torch.FloatTensor(genomic_batch)
                drug_tensor = torch.FloatTensor(drug_fingerprint).unsqueeze(0).repeat(len(genomic_batch), 1)
                
                with torch.no_grad():
                    predictions = model(genomic_tensor, drug_tensor)
                return predictions.numpy()
            
            background = np.zeros((1, len(feature_names)), dtype=np.float32)
            explainer = shap.KernelExplainer(model_predict, background)
            shap_values = explainer.shap_values(genomic_features.reshape(1, -1), nsamples=100)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0][0]
            else:
                shap_values = shap_values[0]
            
            importances = []
            for idx, (feature_name, shap_val) in enumerate(zip(feature_names, shap_values)):
                if abs(shap_val) > 1e-6:
                    importances.append({
                        'Feature': feature_name,
                        'SHAP Value': shap_val,
                        'Impact': abs(shap_val),
                        'Direction': 'Increases PFS' if shap_val > 0 else 'Decreases PFS',
                        'Feature Value': genomic_features[idx]
                    })
        
        # Sort by absolute SHAP value and take top K
        df = pd.DataFrame(importances)
        if not df.empty:
            df = df.sort_values('Impact', ascending=False).head(top_k)
        
        # Create SHAP explanation object (for Deep Learning only)
        shap_explanation = None
        if not is_xgboost and not isinstance(shap_values, list):
            shap_explanation = shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value,
                data=genomic_features,
                feature_names=feature_names
            )
        
        return df, shap_explanation
        
    except Exception as e:
        st.warning(f"⚠️ SHAP calculation failed: {str(e)}. Using fallback method.")
        return get_feature_importance_mock(feature_names, genomic_features, top_k), None


def get_feature_importance_mock(
    feature_names: List[str],
    genomic_features: np.ndarray,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Generate mock feature importance based on non-zero features.
    Fallback when SHAP is not available or fails.
    """
    # Find non-zero features
    nonzero_mask = genomic_features != 0
    nonzero_indices = np.where(nonzero_mask)[0]
    
    if len(nonzero_indices) == 0:
        # All zeros, return empty dataframe
        return pd.DataFrame(columns=['Feature', 'Impact', 'Direction'])
    
    # Create mock importance
    importances = []
    for idx in nonzero_indices:
        if idx < len(feature_names):
            feature_name = feature_names[idx]
            value = genomic_features[idx]
            
            # Mock impact calculation
            impact = abs(value) * np.random.uniform(0.5, 1.5)
            direction = "Increases PFS" if value > 0.5 else "Decreases PFS"
            
            importances.append({
                'Feature': feature_name,
                'Value': value,
                'Impact': impact,
                'Direction': direction
            })
    
    # Sort by impact and take top K
    df = pd.DataFrame(importances)
    if not df.empty:
        df = df.sort_values('Impact', ascending=False).head(top_k)
    
    return df


# ===================================================================
# MAIN APP
# ===================================================================

def main():
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown(
        '<p class="main-header">🧬 Precision Oncology Chemotherapy Sensitivity Predictor</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Predict Progression-Free Survival (PFS) and resistance likelihood using multi-modal deep learning</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p>
        This interactive demo lets you simulate a patient’s chemotherapy course using real-world AACR GENIE NSCLC data.
        You can:
        </p>
        <ul>
            <li>
                Configure <b>chemotherapy regimens</b> by selecting one or more drugs (up to 4) and, where needed, 
                supplying custom SMILES structures.
            </li>
            <li>
                Enter key <b>clinical variables</b> such as age, stage, histology, treatment line, and prior therapies, 
                or leave them blank to use cohort medians.
            </li>
            <li>
                Specify important <b>genomic alterations</b> (e.g., TP53, KRAS, EGFR, ALK and other actionable genes) to see 
                how they may influence predicted outcomes.
            </li>
            <li>
                Obtain a personalized prediction of <b>progression-free survival (PFS)</b> in months and the
                <b>probability of resistance</b> by 6 months, using:
                <ul>
                    <li>XGBoost for <b>first-line</b> treatment (line = 1)</li>
                    <li>Deep learning for <b>later-line</b> treatment (line ≥ 2)</li>
                </ul>
            </li>
            <li>
                Explore <b>feature importance</b> to understand which clinical and genomic factors contributed most to the
                prediction, using SHAP values when available.
            </li>
            <li>
                Compare your simulated patient’s PFS to the <b>training cohort distribution</b> to see whether the predicted
                outcome is better or worse than typical.
            </li>
        </ul>
        <p>
        This is a research tool intended for exploration and hypothesis generation only, not for clinical decision-making.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Load resources
    with st.spinner("Loading stratified models and data..."):
        dl_model, xgb_model, dl_config, xgb_config = load_stratified_models()
        drug_smiles, drug_classes, fp_library, drug_map = load_drug_library()
        feature_names, metadata = load_feature_info()
    
    if dl_model is None and xgb_model is None:
        st.error("❌ No models available. Please train models first.")
        st.info(
            """
            Run these commands to train:
            - `python src/training/train_deep_learning.py` (for previous treatment, line>=2)
            - `python src/training/train_xgboost.py` (for first treatment, line=1)
            """
        )
        st.stop()
        st.stop()
    
    # Sidebar - About section
    with st.sidebar:
        st.header("ℹ️ Stratified Models")
        
        with st.expander("📊 Model Performance", expanded=True):
            st.markdown("**First Treatment (line=1): XGBoost**")
            xgb_metrics = load_classification_metrics(treatment_line=1)
            if xgb_metrics:
                primary_metric = xgb_metrics.get("6 months (PRIMARY - clinical benefit)", {})
                auroc = primary_metric.get('auroc', 0.0)
                st.metric("AUROC", f"{auroc:.3f}", help="XGBoost on 822 samples")
            else:
                st.caption("Metrics not available")
            
            st.markdown("---")
            
            st.markdown("**Previous Treatment (line≥2): Deep Learning**")
            dl_metrics = load_classification_metrics(treatment_line=2)
            if dl_metrics:
                primary_metric = dl_metrics.get("6 months (PRIMARY - clinical benefit)", {})
                auroc = primary_metric.get('auroc', 0.0)
                st.metric("AUROC", f"{auroc:.3f}", help="Deep Learning on 2,112 samples")
            else:
                st.caption("Metrics not available")
        
        with st.expander("🔬 Training Data"):
            st.write("""
            **Dataset:** AACR GENIE NSCLC Cohort
            - **Total Samples:** 2,934 treatment records
            - **Patients:** 1,060 unique
            - **Features:** 1,318 genomic + clinical
            - **Drugs:** 81 chemotherapy agents
            
            **Stratification:**
            - First treatment (line=1): 822 samples
            - Previous treatment (line≥2): 2,112 samples
            """)
        
        with st.expander("🤖 Model Architecture"):
            st.write("""
            **First Treatment: XGBoost**
            - 500 selected features (from 9,510)
            - 5-fold cross-validation
            - Max depth: 4 (shallow trees)
            - Conservative regularization
            
            **Previous Treatment: Deep Learning**
            - Genomic Encoder (3 layers)
            - Drug Fingerprint Encoder (4 layers)
            - Bidirectional Cross-Attention
            - 23M total parameters
            """)
            
        
        with st.expander("📚 Citation"):
            st.write("""            
            Model trained on AACR GENIE data:
            AACR Project GENIE Consortium. (2017). 
            AACR Project GENIE: Powering Precision Medicine 
            through an International Consortium. Cancer Discovery.
            """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Make Prediction", "📈 Feature Importance", "💡 Interpretation Guide"])
    
    with tab1:
        st.header("Patient & Treatment Input")
        # ===================================================================
        # DRUG SELECTION
        # ===================================================================
        st.subheader("Step 1: Chemotherapy Selection, choose a single regimen or up to 4")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Organize drugs by class for display
            drug_options = []
            drug_to_class = {}
            all_actual_drugs = []
            
            for class_name, drugs in sorted(drug_classes.items()):
                if isinstance(drugs, dict):
                    # Nested structure (e.g., Targeted Small-Molecule Inhibitors)
                    for subclass_name, subdrugs in sorted(drugs.items()):
                        drug_options.append(f"─── {class_name} > {subclass_name} ───")
                        for drug in sorted(subdrugs):
                            drug_options.append(f"  • {drug}")
                            drug_to_class[f"  • {drug}"] = f"{class_name} > {subclass_name}"
                            all_actual_drugs.append(f"  • {drug}")
                else:
                    drug_options.append(f"─── {class_name} ───")
                    for drug in sorted(drugs):
                        drug_options.append(f"  • {drug}")
                        drug_to_class[f"  • {drug}"] = class_name
                        all_actual_drugs.append(f"  • {drug}")
            
            # Drug selection
            selected_options = st.multiselect(
                "Select up to 4 drugs (grouped by class)",
                options=drug_options,
                max_selections=4,
                help="Choose 1-4 drugs. Headers (───) are not selectable - only drug names with • can be selected."
            )
            
            # Filter out any accidentally selected headers and extract drug names
            selected_drugs = [opt.replace("  • ", "") for opt in selected_options if opt in all_actual_drugs]
            
            if selected_drugs:
                st.info(f"✅ Selected {len(selected_drugs)} drug(s)")
                for opt in selected_options:
                    if opt in all_actual_drugs:
                        drug = opt.replace("  • ", "")
                        st.caption(f"• {drug} ({drug_to_class.get(opt, 'Unknown')})")
        
        with col2:
            st.write("**Custom SMILES (Optional)**")
            st.caption("If a drug's SMILES is not in our database, you will need to enter it here:")
            
            custom_smiles = {}
            for drug in selected_drugs:
                if drug not in drug_smiles:
                    smiles_input = st.text_input(
                        f"SMILES for {drug}",
                        key=f"smiles_{drug}",
                        placeholder="CCO (example: ethanol)",
                        help="Enter the SMILES notation for this drug"
                    )
                    if smiles_input:
                        custom_smiles[drug] = smiles_input
        
        st.markdown("---")
        
        # ===================================================================
        # PATIENT & tumour VARIABLES
        # ===================================================================
        st.subheader("Step 2: Patient & tumour Variables")
        
        st.info("💡 Leave fields blank to use dataset median values as defaults")
        
        user_inputs = {}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Demographics**")
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=None, placeholder="e.g., 65")
            if age is not None:
                # Assuming age is scaled in training
                user_inputs['age_at_diagnosis_scaled'] = (age - 60) / 15  # Mock scaling
            
            sex = st.selectbox("Sex", ["Unknown", "Male", "Female"])
            if sex != "Unknown":
                user_inputs['sex_Male'] = 1.0 if sex == "Male" else 0.0
                user_inputs['sex_Female'] = 1.0 if sex == "Female" else 0.0
        
        with col2:
            st.write("**tumour Characteristics**")
            
            tumour_type = st.selectbox(
                "Primary tumour Type",
                ["Unknown", "Lung", "Breast", "Colorectal", "Prostate", "Pancreatic", "Other"]
            )
            
            stage = st.selectbox(
                "Stage at Diagnosis",
                ["Unknown", "I", "II", "III", "IV"]
            )
            if stage in ["III", "IV"]:
                user_inputs['stage_advanced'] = 1.0
            
            histology = st.selectbox(
                "Histology",
                ["Unknown", "Adenocarcinoma", "Squamous Cell", "Small Cell", "Other"]
            )
        
        with col3:
            st.write("**Genomic Features**")
            
            mutation_load = st.number_input(
                "tumour Mutation Burden (mutations/Mb)",
                min_value=0.0,
                max_value=100.0,
                value=None,
                placeholder="e.g., 8.5"
            )
            
            st.caption("**Key Mutations** (check if present)")
            st.caption("💡 *Note: EGFR/ALK mutations increase PFS for targeted therapies, but effects vary for chemotherapy*")
            tp53_mut = st.checkbox("TP53 mutation")
            kras_mut = st.checkbox("KRAS mutation")
            egfr_mut = st.checkbox("EGFR mutation")
            alk_mut = st.checkbox("ALK fusion")
            # Additional tracked genes (common in GENIE NSCLC panels)
            st.caption("Additional tracked genes (optional)")
            braf_mut = st.checkbox("BRAF mutation")
            met_mut = st.checkbox("MET alteration (amplification / exon 14 skipping)")
            ros1_fus = st.checkbox("ROS1 fusion")
            ret_fus = st.checkbox("RET fusion")
            erbb2_mut = st.checkbox("ERBB2 (HER2) alteration")
            pik3ca_mut = st.checkbox("PIK3CA mutation")
            pten_mut = st.checkbox("PTEN loss / mutation")
            ntrk_fus = st.checkbox("NTRK fusion")
            brca2_mut = st.checkbox("BRCA2 mutation")
            smad4_mut = st.checkbox("SMAD4 mutation")

            # Map to input features (1.0 if present, 0.0 if absent)
            user_inputs['MUT_BRAF']   = 1.0 if braf_mut else 0.0
            user_inputs['MUT_MET']    = 1.0 if met_mut else 0.0
            user_inputs['MUT_ROS1']   = 1.0 if ros1_fus else 0.0
            user_inputs['MUT_RET']    = 1.0 if ret_fus else 0.0
            user_inputs['MUT_ERBB2']  = 1.0 if erbb2_mut else 0.0
            user_inputs['MUT_PIK3CA'] = 1.0 if pik3ca_mut else 0.0
            user_inputs['MUT_PTEN']   = 1.0 if pten_mut else 0.0
            user_inputs['MUT_NTRK']   = 1.0 if ntrk_fus else 0.0
            user_inputs['MUT_BRCA2']  = 1.0 if brca2_mut else 0.0
            user_inputs['MUT_SMAD4']  = 1.0 if smad4_mut else 0.0
            
            # Explicitly set mutation values (1.0 if checked, 0.0 if not)
            user_inputs['MUT_TP53'] = 1.0 if tp53_mut else 0.0
            user_inputs['MUT_KRAS'] = 1.0 if kras_mut else 0.0
            user_inputs['MUT_EGFR'] = 1.0 if egfr_mut else 0.0
            user_inputs['MUT_ALK'] = 1.0 if alk_mut else 0.0
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Treatment History**")
            treatment_line = st.number_input(
                "Treatment Line",
                min_value=1,
                max_value=10,
                value=1,
                help="1 = first-line, 2 = second-line, etc."
            )
            user_inputs['treatment_line'] = float(treatment_line)
            
            prior_chemo = st.checkbox("Prior chemotherapy")
            prior_immunotherapy = st.checkbox("Prior immunotherapy")
            
            # Explicitly set prior treatment values
            user_inputs['prior_chemotherapy'] = 1.0 if prior_chemo else 0.0
            user_inputs['prior_immunotherapy'] = 1.0 if prior_immunotherapy else 0.0
        
        with col2:
            st.write("**Performance Status**")
            ecog = st.selectbox(
                "ECOG Performance Status",
                ["Unknown", "0 (Fully active)", "1 (Restricted)", "2 (Ambulatory)", "3 (Limited)", "4 (Disabled)"]
            )
            
        # ===================================================================
        # PREDICTION
        # ===================================================================
        st.markdown("---")
        st.subheader("Step 3: Generate Prediction")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            predict_button = st.button("🔮 Predict PFS & Resistance", type="primary", use_container_width=True)
        
        if predict_button:
            if not selected_drugs:
                st.error("❌ Please select at least one drug")
            else:
                with st.spinner("Running model inference..."):
                    # Automatically set drug-related features based on selection
                    user_inputs['drugs_in_regimen'] = float(len(selected_drugs))
                    user_inputs['is_combination'] = float(1 if len(selected_drugs) > 1 else 0)
                    
                    # Create drug fingerprint
                    drug_fp = create_drug_fingerprint_vector(
                        selected_drugs,
                        drug_smiles,
                        custom_smiles,
                        fp_library,
                        drug_map
                    )
                    
                    # Create genomic feature vector
                    genomic_features = create_genomic_feature_vector(
                        feature_names,
                        user_inputs,
                        metadata
                    )
                    
                    # Predict using appropriate stratified model
                    pfs_months, resistance_prob, log_pred, confidence, model_used = predict_pfs_stratified(
                        dl_model,
                        xgb_model,
                        genomic_features,
                        drug_fp,
                        treatment_line,
                        xgb_config
                    )
                    
                    # Store in session state for other tabs
                    st.session_state['last_prediction'] = {
                        'pfs_months': pfs_months,
                        'resistance_prob': resistance_prob,
                        'log_pred': log_pred,
                        'confidence': confidence,
                        'genomic_features': genomic_features,
                        'drug_fingerprint': drug_fp,
                        'selected_drugs': selected_drugs,
                        'treatment_line': treatment_line,
                        'model_used': model_used
                    }
                
                # Display results
                st.success(f"✅ Prediction Complete! (Using {model_used})")
                
                # Debug: Show input summary
                with st.expander("🔍 Input Summary (Click to verify)", expanded=False):
                    st.write(f"**Treatment Line:** {treatment_line}")
                    st.write(f"**Drugs:** {', '.join(selected_drugs)}")
                    st.write(f"**Mutations:**")
                    st.write(f"  - TP53: {'✓ Present (1.0)' if user_inputs.get('MUT_TP53', 0) == 1.0 else '✗ Absent (0.0)'}")
                    st.write(f"  - KRAS: {'✓ Present (1.0)' if user_inputs.get('MUT_KRAS', 0) == 1.0 else '✗ Absent (0.0)'}")
                    st.write(f"  - EGFR: {'✓ Present (1.0)' if user_inputs.get('MUT_EGFR', 0) == 1.0 else '✗ Absent (0.0)'}")
                    st.write(f"  - ALK: {'✓ Present (1.0)' if user_inputs.get('MUT_ALK', 0) == 1.0 else '✗ Absent (0.0)'}")
                    
                    # Check if EGFR is in selected features (for XGBoost)
                    if treatment_line == 1 and xgb_config.get('selected_features') is not None:
                        selected_features = xgb_config['selected_features']
                        # Find EGFR mutation index
                        try:
                            egfr_idx = feature_names.index('MUT_EGFR')
                            if egfr_idx in selected_features:
                                st.success("✅ MUT_EGFR is in XGBoost's 500 selected features")
                            else:
                                st.warning("⚠️ MUT_EGFR was NOT selected by XGBoost (not in top 500 features)")
                                st.caption("This explains why EGFR mutation status doesn't affect predictions")
                        except ValueError:
                            st.info("MUT_EGFR not found in feature list")
                
                # Model-specific insights
                if treatment_line == 1:
                    st.info("""
                    **First Treatment Model (XGBoost)**: 
                    - Trained on 822 first-line treatment records
                    - Uses 500 selected features (from 9,510)
                    - 5-fold cross-validated (AUROC = 0.703)
                    - Optimized for small dataset with aggressive feature selection
                    """)
                else:
                    st.info(f"""
                    **Previous Treatment Model (Deep Learning)**: 
                    - Trained on 2,112 records from treatment lines 2+
                    - 23M parameters with cross-attention
                    - Captures complex drug-genomic interactions
                    - Treatment line {treatment_line} (later lines typically show worse outcomes)
                    """)
                
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Predicted PFS",
                        value=f"{pfs_months:.1f} months",
                        delta=None
                    )
                    
                    # Interpretation
                    if pfs_months >= 6:
                        st.success("✅ Likely responder (PFS ≥ 6 months)")
                    elif pfs_months >= 3:
                        st.warning("⚠️ Intermediate response (3-6 months)")
                    else:
                        st.error("⚠️ Likely poor response (PFS < 3 months)")
                
                with col2:
                    st.metric(
                        label="Resistance Risk at 6 Months",
                        value=f"{resistance_prob:.1f}%",
                        delta=None,
                        delta_color="inverse"
                    )
                    
                    # Progress bar
                    st.progress(resistance_prob / 100)
                    
                    if resistance_prob < 30:
                        st.success("🟢 Low risk")
                    elif resistance_prob < 60:
                        st.warning("🟡 Moderate risk")
                    else:
                        st.error("🔴 High risk")
                
                with col3:
                    st.metric(
                        label="Confidence Score",
                        value=f"{confidence:.2f}",
                        help="Confidence in classification (0-1). Based on distance from 6-month threshold."
                    )
                    
                    # Confidence interpretation
                    if confidence >= 0.7:
                        st.success("🟢 High confidence")
                    elif confidence >= 0.4:
                        st.warning("🟡 Moderate confidence")
                    else:
                        st.info("🔵 Low confidence (near decision boundary)")
                    
                    # Show regimen
                    st.caption("**Treatment Regimen:**")
                    for drug in selected_drugs:
                        st.caption(f"• {drug}")
                
                # Visual comparison
                st.markdown("---")
                st.subheader("📊 Comparison to Cohort")
                
                # Create comparison chart
                cohort_median = 2.53  # From your data
                cohort_q25 = 1.2
                cohort_q75 = 5.8
                
                fig = go.Figure()
                
                # Cohort distribution
                fig.add_trace(go.Box(
                    y=[cohort_q25, cohort_median, cohort_q75],
                    name="Cohort Distribution",
                    boxmean='sd',
                    marker_color='lightblue'
                ))
                
                # Patient prediction
                fig.add_trace(go.Scatter(
                    x=['Prediction'],
                    y=[pfs_months],
                    mode='markers',
                    name='Your Prediction',
                    marker=dict(size=20, color='red', symbol='star')
                ))
                
                fig.update_layout(
                    title="Predicted PFS vs. Cohort Distribution",
                    yaxis_title="PFS (months)",
                    showlegend=True,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📈 Feature Importance Analysis")
        
        if 'last_prediction' not in st.session_state:
            st.info("👈 Make a prediction first to see feature importance")
        else:
            pred_data = st.session_state['last_prediction']
            genomic_features = pred_data['genomic_features']
            drug_fingerprint = pred_data['drug_fingerprint']
            treatment_line = pred_data.get('treatment_line', 1)
            model_used = pred_data.get('model_used', 'Unknown')
            
            st.info(f"**Model Used:** {model_used}")
            
            # Determine which model and whether it's XGBoost
            is_xgboost = (treatment_line == 1)
            active_model = xgb_model if is_xgboost else dl_model
            
            if active_model is None:
                st.error("Model not available for feature importance analysis")
            else:
                st.subheader("Top Contributing Features")
                
                # Choose analysis method
                analysis_method = st.radio(
                    "Analysis Method:",
                    options=["SHAP (Accurate)", "Quick Estimate"],
                    horizontal=True,
                    help="SHAP provides accurate feature attributions. Quick estimate is instant."
                )
                
                if analysis_method == "SHAP (Accurate)" and HAS_SHAP:
                    with st.spinner("Calculating SHAP values (this may take 30-60 seconds)..."):
                        importance_df, shap_explanation = get_feature_importance_shap(
                            active_model,
                            genomic_features,
                            drug_fingerprint,
                            feature_names,
                            top_k=15,
                            is_xgboost=is_xgboost,
                            xgb_config=xgb_config if is_xgboost else None
                        )
                else:
                    importance_df = get_feature_importance_mock(feature_names, genomic_features, top_k=15)
                    shap_explanation = None
                
            if not importance_df.empty:
                # Bar plot
                if 'SHAP Value' in importance_df.columns:
                    # SHAP-based plot
                    fig = px.bar(
                        importance_df,
                        x='SHAP Value',
                        y='Feature',
                        orientation='h',
                        color='SHAP Value',
                        title=f"Top 15 Features by SHAP Value ({model_used})",
                        labels={'SHAP Value': 'SHAP Value (log-PFS)', 'Feature': ''},
                        color_continuous_scale='RdBu_r',
                        color_continuous_midpoint=0
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Mock importance plot (when SHAP not available)
                    fig = px.bar(
                        importance_df,
                        x='Impact',
                        y='Feature',
                        orientation='h',
                        color='Direction',
                        title="Top 15 Features by Estimated Impact",
                        labels={'Impact': 'Impact on Prediction', 'Feature': ''},
                        color_discrete_map={
                            'Increases PFS': '#2ca02c',
                            'Decreases PFS': '#d62728'
                        }
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                # SHAP waterfall plot
                if shap_explanation is not None and HAS_SHAP:
                    st.subheader("SHAP Waterfall Plot")
                    st.caption("Shows how each feature pushes the prediction from the base value")
                    
                    try:
                        import matplotlib.pyplot as plt
                        fig_waterfall, ax = plt.subplots(figsize=(10, 8))
                        shap.plots.waterfall(shap_explanation, max_display=15, show=False)
                        st.pyplot(fig_waterfall, use_container_width=True)
                        plt.close()
                    except Exception as e:
                        st.warning(f"Could not generate waterfall plot: {str(e)}")
                
                # Table
                st.subheader("Feature Details")
                available_cols = importance_df.columns.tolist()
                display_cols = [col for col in ['Feature', 'SHAP Value', 'Feature Value', 'Direction'] if col in available_cols]
                st.dataframe(
                    importance_df[display_cols] if display_cols else importance_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Explanation
                st.info("""
                **SHAP Value Interpretation:**
                - **Positive SHAP values** (red): Feature increases predicted PFS
                - **Negative SHAP values** (blue): Feature decreases predicted PFS
                - **Magnitude**: How much the feature changes the prediction (in log-space)
                """)
                
                if analysis_method != "SHAP (Accurate)" or not HAS_SHAP:
                    st.warning("⚠️ Using quick estimate. For accurate feature importance, select 'SHAP (Accurate)' above.")
    
    with tab3:
        st.header("💡 How to Interpret Predictions")
        
        st.markdown("""
        ### Understanding Your Results
        
        #### Predicted PFS (Progression-Free Survival)
        - **Definition:** Expected time until disease progression or death
        - **Clinical Thresholds:**
            - ≥6 months: Clinical benefit (responsive to treatment)
            - 3-6 months: Intermediate response
            - <3 months: Poor response (consider alternative therapy)
        
        #### Resistance Probability
        - **Definition:** Likelihood of developing resistance by 6 months
        - **Risk Levels:**
            - <30%: Low risk (favorable prognosis)
            - 30-60%: Moderate risk (monitor closely)
            - >60%: High risk (consider combination therapy or alternatives)
        
        ### Stratified Model Approach
        
        This system uses **two specialized models** based on treatment history:
        
        #### First Treatment (line = 1): XGBoost
        - **Dataset:** 822 first-line treatment records
        - **Performance:** AUROC = 0.703
        - **Features:** 500 selected (from 9,510 total)
        - **Why XGBoost?** Small dataset requires simpler model with aggressive feature selection
        - **Strengths:** Robust to noise, interpretable feature importance
        
        #### Previous Treatment (line ≥ 2): Deep Learning
        - **Dataset:** 2,112 later-line treatment records
        - **Performance:** AUROC ≈ 0.70-0.75 (expected)
        - **Features:** All 9,510 features with 23M parameters
        - **Why Deep Learning?** Sufficient data for complex drug-genomic interactions
        - **Strengths:** Cross-attention captures non-linear relationships
        
        ### Important Limitations
        
        ⚠️ **This is a research tool, not for clinical decisions**
        
        - Predictions based on historical NSCLC cohort (AACR GENIE)
        - May not generalize to all cancer types or populations
        - Model does not account for:
            - Patient comorbidities
            - Drug interactions
            - Treatment compliance
            - Evolving tumour biology
            - Quality of life factors
        - **Always consult with oncologists for treatment decisions**
        
        ### Feature Importance Caveats
        
        - SHAP values show association, not causation
        - Treatment line is often the strongest predictor (clinical reality)
        - Genomic alterations have complex interactions
        - Drug selection effects may be confounded by indication
        
        ### Model Architecture Details
        
        **XGBoost (First Treatment):**
        - Gradient boosting with shallow trees (max depth = 4)
        - 5-fold cross-validation for hyperparameter tuning
        - Conservative regularization (L1=0.5, L2=1.0)
        - Feature selection based on gain importance
        
        **Deep Learning (Previous Treatment):**
        1. **Genomic Encoder:** 3-layer MLP processes 1,318 features
        2. **Drug Encoder:** 4-layer MLP processes 8,192-dim fingerprints
        3. **Cross-Attention:** Bidirectional attention between drugs and genomics
        4. **Output Head:** Predicts log(PFS+1) for continuous estimation
        
        ### Training Details
        
        **Data Split:** Patient-level 70/15/15 (prevents data leakage)
        - All treatments from same patient stay in same split
        - Stratified by patient median PFS for balanced outcomes
        
        **Regularization:**
        - Dropout: 0.5-0.6 (genomic), 0.4-0.5 (drug), 0.5-0.6 (head)
        - Early stopping: patience = 10-15 epochs
        - Weight decay (L2): 1e-3 to 5e-3
        
        ### Citation & Contact            
        **Data Source:** The AACR Project GENIE Consortium. AACR Project GENIE: Powering Precision Medicine Through An International Consortium, Cancer Discov. 2017 Aug;7(8):818-831 
        
        **Data Version:** GENIE NSCLC 2.0-public
        
        **Model:** Stratified prediction for treatment line-specific outcomes
                     """)
        
        st.markdown("---")
        
        with st.expander("🔬 Technical Implementation"):
            st.write(f"""
            **First Treatment Model (XGBoost):**
            - Training samples: 822
            - Selected features: 500 (from 9,510)
            - CV folds: 5
            - Best params: depth=4, lr=0.03, trees=100
            
            **Previous Treatment Model (Deep Learning):**
            - Training samples: 2,112
            - Total parameters: ~23M
            - Embedding dim: {dl_config.get('embed_dim', 256)}
            - Training epochs: {dl_config.get('num_epochs', 100)}
            - Best epoch: {dl_config.get('best_epoch', 'N/A')}
            """)
    
    # Footer
    st.markdown("---")
    st.caption("Precision Oncology Chemotherapy Sensitivity Predictor | Powered by PyTorch & Streamlit")


if __name__ == "__main__":
    main()
