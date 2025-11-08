"""
Prepare Clean Data for Training
================================

Loads the artifacts from feature_engineering.ipynb and prepares them for model training.
The workflow is: dataprocess.ipynb → feature_engineering.ipynb → tonumpy.py → train.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def main():
    print("=" * 70)
    print("PREPARING DATA FOR TRAINING")
    print("=" * 70)
    
    # Load artifacts from feature_engineering.ipynb
    print("\nLoading artifacts from feature_engineering.ipynb...")
    
    try:
        # Load tabular features (genomics, clinical, microbiome)
        X_tabular = np.load('artifacts/X_tabular.npy')
        print(f"✓ X_tabular.npy: {X_tabular.shape}")
        
        # Load drug fingerprints
        X_drug_fp = np.load('artifacts/X_drug_fp.npy')
        print(f"✓ X_drug_fp.npy: {X_drug_fp.shape}")
        
        # Load target variable
        y_effectiveness = np.load('artifacts/y_effectiveness.npy')
        print(f"✓ y_effectiveness.npy: {y_effectiveness.shape}")
        
        # Load metadata for patient IDs
        metadata = pd.read_csv('artifacts/sample_metadata.csv')
        patient_ids = metadata['Patient_ID'].values
        print(f"✓ sample_metadata.csv: {metadata.shape}")
        
        # Load feature names
        with open('artifacts/feature_names.json', 'r') as f:
            feature_info = json.load(f)
        print(f"✓ feature_names.json")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Missing file - {e}")
        print("\n→ You need to run feature_engineering.ipynb first!")
        print("  Specifically, run all cells to generate artifacts/")
        return
    
    # Verify dimensions match
    n_samples = len(X_tabular)
    assert len(X_drug_fp) == n_samples, f"Drug FP mismatch: {len(X_drug_fp)} vs {n_samples}"
    assert len(y_effectiveness) == n_samples, f"Target mismatch: {len(y_effectiveness)} vs {n_samples}"
    assert len(patient_ids) == n_samples, f"Patient ID mismatch: {len(patient_ids)} vs {n_samples}"
    
    print(f"\n✓ All dimensions match: {n_samples} samples")
    
    # Filter to valid PFS samples (if using PFS as target instead of effectiveness_score)
    # Load PFS from original dataset
    ml_data = pd.read_csv('data/processed/ml_features_clean.csv', low_memory=False)
    targets_df = pd.read_csv('data/processed/ml_targets.csv', low_memory=False)
    
    valid_mask = targets_df['pfs_months'].notna().values
    print(f"\nFiltering to valid PFS samples...")
    print(f"  Before: {n_samples} samples")
    print(f"  Valid PFS: {valid_mask.sum()} samples")
    
    # Apply filter
    X_tabular_clean = X_tabular[valid_mask]
    X_drug_fp_clean = X_drug_fp[valid_mask]
    patient_ids_clean = patient_ids[valid_mask]
    
    # Use log-transformed PFS as target (more standard than effectiveness_score)
    y_pfs = np.log1p(targets_df['pfs_months'].values[valid_mask]).astype(np.float32)
    
    print(f"  After: {len(X_tabular_clean)} samples")
    print(f"  After: {len(X_tabular_clean)} samples")
    
    # Save cleaned data for training
    output_dir = Path('artifacts')
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / 'X_tabular_clean.npy', X_tabular_clean)
    np.save(output_dir / 'X_drug_fp_clean.npy', X_drug_fp_clean)
    np.save(output_dir / 'y_clean.npy', y_pfs)
    np.save(output_dir / 'patient_ids_clean.npy', patient_ids_clean)
    
    # Save feature names
    with open(output_dir / 'feature_names_clean.txt', 'w') as f:
        f.write(f"Cleaned Training Data Summary\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Total samples: {len(y_pfs)}\n")
        f.write(f"Unique patients: {len(np.unique(patient_ids_clean))}\n\n")
        
        f.write(f"Tabular Features: {X_tabular_clean.shape[1]}\n")
        f.write(f"  (from feature_names.json)\n")
        for feat_name in feature_info['tabular_features'][:20]:
            f.write(f"    {feat_name}\n")
        if len(feature_info['tabular_features']) > 20:
            f.write(f"    ... and {len(feature_info['tabular_features']) - 20} more\n")
        
        f.write(f"\nDrug Fingerprints: {X_drug_fp_clean.shape[1]} dimensions\n")
        f.write(f"  Morgan fingerprints (radius=2, nBits=2048) × 4 drug positions\n\n")
        
        f.write(f"Total Features: {X_tabular_clean.shape[1] + X_drug_fp_clean.shape[1]}\n")
        f.write(f"Target: log(PFS_months + 1)\n")
    
    print(f"\n✓ Saved cleaned datasets to {output_dir}/")
    print(f"  - X_tabular_clean.npy: {X_tabular_clean.shape}")
    print(f"  - X_drug_fp_clean.npy: {X_drug_fp_clean.shape}")
    print(f"  - y_clean.npy: {y_pfs.shape}")
    print(f"  - patient_ids_clean.npy: {patient_ids_clean.shape}")
    print(f"  - feature_names_clean.txt")
    
    # Summary statistics
    print(f"\n" + "=" * 70)
    print(f"DATA SUMMARY")
    print(f"=" * 70)
    print(f"  Samples: {len(y_pfs)}")
    print(f"  Patients: {len(np.unique(patient_ids_clean))}")
    print(f"  Tabular features: {X_tabular_clean.shape[1]}")
    print(f"  Drug features: {X_drug_fp_clean.shape[1]}")
    print(f"  Total features: {X_tabular_clean.shape[1] + X_drug_fp_clean.shape[1]}")
    print(f"  Target (log PFS): [{y_pfs.min():.3f}, {y_pfs.max():.3f}]")
    print(f"  Target mean: {y_pfs.mean():.3f} ± {y_pfs.std():.3f}")
    print("=" * 70)
    print("\n✓ Ready for training! Run train.py next.")

if __name__ == "__main__":
    main()
