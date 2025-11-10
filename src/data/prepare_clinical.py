"""
Prepare clinical features from the dataset for the enhanced GNN model
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

def prepare_clinical_features():
    """
    Extract and encode clinical variables from the dataset
    """
    print("Loading dataset...")
    df = pd.read_csv(DATA_DIR / "ml_dataset_complete.csv")
    
    print(f"  Total samples: {len(df)}")
    
    # Select clinical features
    clinical_features = [
        'age_at_diagnosis',      # Continuous
        'sex',                   # Categorical (0/1)
        'treatment_line',        # Ordinal (1, 2, 3, ...)
        'tumor_grade',           # Ordinal
        'stage_at_diagnosis',    # Ordinal/Categorical
        'pre_treatment_stage',   # Ordinal/Categorical  
        'smoking_history',       # Categorical
        'distant_mets'           # Binary (0/1)
    ]
    
    print(f"\nExtracting clinical features...")
    for feat in clinical_features:
        print(f"  {feat}: {df[feat].dtype}, missing={df[feat].isna().sum()}")
    
    # Create clinical feature matrix
    X_clinical = df[clinical_features].copy()
    
    # Encoding strategy
    print("\nEncoding clinical features...")
    
    # 1. Age - already numeric, just fill missing
    if X_clinical['age_at_diagnosis'].isna().any():
        median_age = X_clinical['age_at_diagnosis'].median()
        X_clinical['age_at_diagnosis'].fillna(median_age, inplace=True)
        print(f"  age_at_diagnosis: filled {X_clinical['age_at_diagnosis'].isna().sum()} missing with median {median_age:.1f}")
    
    # 2. Sex - convert to binary (0/1)
    if X_clinical['sex'].dtype == 'object':
        le_sex = LabelEncoder()
        X_clinical['sex'] = le_sex.fit_transform(X_clinical['sex'].fillna('Unknown'))
        print(f"  sex: encoded as {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
    
    # 3. Treatment line - already numeric, fill missing with 1 (first line)
    X_clinical['treatment_line'].fillna(1, inplace=True)
    
    # 4. Tumor grade - encode ordinally
    if X_clinical['tumor_grade'].dtype == 'object':
        # Common grades: Grade I, II, III, IV, etc.
        grade_mapping = {
            'Grade I': 1, 'Grade II': 2, 'Grade III': 3, 'Grade IV': 4,
            'Grade 1': 1, 'Grade 2': 2, 'Grade 3': 3, 'Grade 4': 4,
            'I': 1, 'II': 2, 'III': 3, 'IV': 4,
            '1': 1, '2': 2, '3': 3, '4': 4,
            'Unknown': 0, 'Not applicable': 0
        }
        X_clinical['tumor_grade'] = X_clinical['tumor_grade'].map(grade_mapping).fillna(0)
        print(f"  tumor_grade: mapped to ordinal [0-4]")
    else:
        X_clinical['tumor_grade'].fillna(0, inplace=True)
    
    # 5. Stage at diagnosis - encode ordinally
    if X_clinical['stage_at_diagnosis'].dtype == 'object':
        stage_mapping = {
            'Stage 0': 0, 'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IV': 4,
            'Stage IA': 1, 'Stage IB': 1, 'Stage IIA': 2, 'Stage IIB': 2,
            'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
            'Stage IVA': 4, 'Stage IVB': 4,
            '0': 0, 'I': 1, 'II': 2, 'III': 3, 'IV': 4,
            'Unknown': -1, 'Not applicable': -1
        }
        X_clinical['stage_at_diagnosis'] = X_clinical['stage_at_diagnosis'].map(stage_mapping).fillna(-1)
        print(f"  stage_at_diagnosis: mapped to ordinal [-1 to 4]")
    else:
        X_clinical['stage_at_diagnosis'].fillna(-1, inplace=True)
    
    # 6. Pre-treatment stage - same as stage
    if X_clinical['pre_treatment_stage'].dtype == 'object':
        X_clinical['pre_treatment_stage'] = X_clinical['pre_treatment_stage'].map(stage_mapping).fillna(-1)
        print(f"  pre_treatment_stage: mapped to ordinal [-1 to 4]")
    else:
        X_clinical['pre_treatment_stage'].fillna(-1, inplace=True)
    
    # 7. Smoking history - encode as ordinal (never, former, current)
    if X_clinical['smoking_history'].dtype == 'object':
        smoking_mapping = {
            'Never': 0, 'Former': 1, 'Current': 2,
            'Non-smoker': 0, 'Ex-smoker': 1, 'Smoker': 2,
            'Unknown': -1, 'Not applicable': -1
        }
        X_clinical['smoking_history'] = X_clinical['smoking_history'].map(smoking_mapping).fillna(-1)
        print(f"  smoking_history: mapped to ordinal [-1 to 2]")
    else:
        X_clinical['smoking_history'].fillna(-1, inplace=True)
    
    # 8. Distant metastases - convert to binary
    if X_clinical['distant_mets'].dtype == 'object':
        mets_mapping = {
            'Yes': 1, 'No': 0, 'yes': 1, 'no': 0,
            'Y': 1, 'N': 0, 'True': 1, 'False': 0,
            '1': 1, '0': 0, 1: 1, 0: 0,
            'Unknown': 0, 'Not applicable': 0
        }
        X_clinical['distant_mets'] = X_clinical['distant_mets'].map(mets_mapping).fillna(0)
        print(f"  distant_mets: mapped to binary (0/1)")
    else:
        X_clinical['distant_mets'].fillna(0, inplace=True)
    
    # Convert to numpy array
    X_clinical_array = X_clinical.values.astype(np.float32)
    
    # Normalize continuous features (age, treatment_line)
    scaler = StandardScaler()
    # Only normalize age (column 0) and treatment_line (column 2)
    X_clinical_array[:, [0, 2]] = scaler.fit_transform(X_clinical_array[:, [0, 2]])
    
    print(f"\nClinical features prepared:")
    print(f"  Shape: {X_clinical_array.shape}")
    print(f"  Range: [{X_clinical_array.min():.2f}, {X_clinical_array.max():.2f}]")
    print(f"  Mean: {X_clinical_array.mean():.2f}, Std: {X_clinical_array.std():.2f}")
    
    # Check for NaN
    if np.isnan(X_clinical_array).any():
        print(f"  WARNING: {np.isnan(X_clinical_array).sum()} NaN values found!")
        # Fill any remaining NaN with 0
        X_clinical_array = np.nan_to_num(X_clinical_array, 0)
    
    # Save
    output_file = ARTIFACTS_DIR / "X_clinical_clean.npy"
    np.save(output_file, X_clinical_array)
    print(f"\nSaved to: {output_file}")
    
    # Also save feature names and scaler for reference
    metadata = {
        'feature_names': clinical_features,
        'scaler': scaler,
        'shape': X_clinical_array.shape,
        'description': {
            'age_at_diagnosis': 'Patient age at diagnosis (normalized)',
            'sex': 'Binary encoding of sex',
            'treatment_line': 'Treatment line number (normalized)',
            'tumor_grade': 'Tumor grade (0-4)',
            'stage_at_diagnosis': 'Stage at diagnosis (-1 to 4)',
            'pre_treatment_stage': 'Pre-treatment stage (-1 to 4)',
            'smoking_history': 'Smoking history (-1=unknown, 0=never, 1=former, 2=current)',
            'distant_mets': 'Presence of distant metastases (0/1)'
        }
    }
    
    with open(ARTIFACTS_DIR / "clinical_metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to: {ARTIFACTS_DIR / 'clinical_metadata.pkl'}")
    
    return X_clinical_array

if __name__ == "__main__":
    X_clinical = prepare_clinical_features()
    print("\nDone!")
