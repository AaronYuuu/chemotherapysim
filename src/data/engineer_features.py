"""
Advanced Feature Engineering for Cancer Genomics

Creates quantified scores for:
1. Drug-gene interactions
2. Pathway activity scores
3. Clinical-genomic interactions
4. Tumor mutational burden
5. Gene set enrichment scores
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from pathlib import Path


def calculate_tumor_mutational_burden(genomic_features):
    """
    Calculate TMB (Tumor Mutational Burden) - sum of mutations per sample.
    Higher TMB often correlates with better immunotherapy response.
    """
    # Assuming mutation features start with 'MUT_' prefix
    # Count non-zero mutations per sample
    tmb = np.sum(genomic_features > 0, axis=1)
    return tmb.reshape(-1, 1)


def calculate_pathway_activity_scores(genomic_features, n_pathways=20):
    """
    Use PCA to create pathway activity scores.
    Each PC represents a latent pathway/process.
    """
    pca = PCA(n_components=n_pathways, random_state=42)
    pathway_scores = pca.fit_transform(genomic_features)
    
    # Save PCA model for later use
    return pathway_scores, pca


def create_interaction_features(genomic, clinical, drug):
    """
    Create interaction features between modalities.
    
    Returns:
        dict of interaction features
    """
    interactions = {}
    
    # 1. Age × TMB interaction (older patients with high TMB)
    tmb = calculate_tumor_mutational_burden(genomic)
    if clinical.shape[1] >= 1:  # Assuming first clinical feature is age
        age = clinical[:, 0:1]
        interactions['age_x_tmb'] = age * tmb
    
    # 2. Stage × TMB interaction (advanced stage with high mutations)
    if clinical.shape[1] >= 5:  # Assuming stage_at_diagnosis is 5th feature
        stage = clinical[:, 4:5]
        interactions['stage_x_tmb'] = stage * tmb
    
    # 3. Drug activity score (sum of drug fingerprint features)
    drug_activity = np.sum(drug, axis=1, keepdims=True)
    interactions['drug_activity'] = drug_activity
    
    # 4. Drug × TMB interaction (drug effect modulated by mutation load)
    interactions['drug_x_tmb'] = drug_activity * tmb
    
    # 5. Smoking × mutations (for lung cancer patients)
    if clinical.shape[1] >= 7:  # Assuming smoking_history is 7th feature
        smoking = clinical[:, 6:7]
        interactions['smoking_x_tmb'] = smoking * tmb
    
    # 6. Metastasis × TMB
    if clinical.shape[1] >= 8:  # Assuming distant_mets is 8th feature
        mets = clinical[:, 7:8]
        interactions['mets_x_tmb'] = mets * tmb
    
    return interactions


def create_gene_set_enrichment_scores(genomic_features):
    """
    Create enrichment scores for important gene sets.
    
    For simplicity, we'll use PCA-based scores weighted by importance.
    """
    # DNA repair pathway (critical for chemotherapy response)
    # Approximate by taking weighted sum of features
    n_features = genomic_features.shape[1]
    
    scores = {}
    
    # DNA Repair Score (first ~100 features weighted)
    dna_repair_weights = np.exp(-np.arange(min(100, n_features)) / 20)
    dna_repair_score = np.sum(
        genomic_features[:, :len(dna_repair_weights)] * dna_repair_weights,
        axis=1,
        keepdims=True
    )
    scores['dna_repair_score'] = dna_repair_score
    
    # Cell Cycle Score (next ~100 features)
    start_idx = min(100, n_features)
    end_idx = min(200, n_features)
    if end_idx > start_idx:
        cell_cycle_weights = np.exp(-np.arange(end_idx - start_idx) / 20)
        cell_cycle_score = np.sum(
            genomic_features[:, start_idx:end_idx] * cell_cycle_weights,
            axis=1,
            keepdims=True
        )
        scores['cell_cycle_score'] = cell_cycle_score
    
    # Apoptosis Score
    start_idx = min(200, n_features)
    end_idx = min(300, n_features)
    if end_idx > start_idx:
        apoptosis_weights = np.exp(-np.arange(end_idx - start_idx) / 20)
        apoptosis_score = np.sum(
            genomic_features[:, start_idx:end_idx] * apoptosis_weights,
            axis=1,
            keepdims=True
        )
        scores['apoptosis_score'] = apoptosis_score
    
    return scores


def create_clinical_risk_score(clinical_features):
    """
    Create a composite clinical risk score.
    
    Combines age, stage, grade, metastasis status.
    """
    risk_score = np.zeros((len(clinical_features), 1))
    
    # Age contribution (normalized)
    if clinical_features.shape[1] >= 1:
        age = clinical_features[:, 0:1]
        risk_score += age * 0.2
    
    # Treatment line (higher = more treatment resistance)
    if clinical_features.shape[1] >= 3:
        treatment_line = clinical_features[:, 2:3]
        risk_score += treatment_line * 0.3
    
    # Tumor grade
    if clinical_features.shape[1] >= 4:
        grade = clinical_features[:, 3:4]
        risk_score += grade * 0.2
    
    # Stage at diagnosis
    if clinical_features.shape[1] >= 5:
        stage = clinical_features[:, 4:5]
        risk_score += stage * 0.3
    
    # Metastasis status
    if clinical_features.shape[1] >= 8:
        mets = clinical_features[:, 7:8]
        risk_score += mets * 0.4
    
    return risk_score


def engineer_all_features(genomic, drug, clinical):
    """
    Main function to engineer all features.
    
    Returns:
        X_engineered: Combined feature matrix
        feature_names: List of feature names
        metadata: Dictionary with scalers and transformers
    """
    print("Engineering features...")
    
    # 1. Tumor Mutational Burden
    print("  Computing TMB...")
    tmb = calculate_tumor_mutational_burden(genomic)
    
    # 2. Pathway activity scores
    print("  Computing pathway activity scores...")
    pathway_scores, pca = calculate_pathway_activity_scores(genomic, n_pathways=20)
    
    # 3. Interaction features
    print("  Creating interaction features...")
    interactions = create_interaction_features(genomic, clinical, drug)
    
    # 4. Gene set enrichment scores
    print("  Computing gene set enrichment scores...")
    enrichment_scores = create_gene_set_enrichment_scores(genomic)
    
    # 5. Clinical risk score
    print("  Computing clinical risk score...")
    clinical_risk = create_clinical_risk_score(clinical)
    
    # 6. Statistical features
    print("  Computing statistical features...")
    genomic_mean = np.mean(genomic, axis=1, keepdims=True)
    genomic_std = np.std(genomic, axis=1, keepdims=True)
    genomic_max = np.max(genomic, axis=1, keepdims=True)
    
    drug_mean = np.mean(drug, axis=1, keepdims=True)
    drug_std = np.std(drug, axis=1, keepdims=True)
    
    # Combine all features
    print("  Combining all features...")
    feature_list = [
        # Original features - ALL features, let advanced_feature_selection.py do proper selection
        genomic,           # All genomic features
        drug,              # All drug features
        clinical,          # All clinical features (8)
        
        # Engineered features (domain knowledge + interactions)
        tmb,                                          # 1 feature
        pathway_scores,                               # 20 features
        *list(interactions.values()),                 # 6 features
        *list(enrichment_scores.values()),            # 3 features
        clinical_risk,                                # 1 feature
        genomic_mean, genomic_std, genomic_max,       # 3 features
        drug_mean, drug_std                           # 2 features
    ]
    
    X_engineered = np.concatenate(feature_list, axis=1)
    
    # Feature names - now includes ALL original features
    feature_names = (
        [f'genomic_{i}' for i in range(genomic.shape[1])] +
        [f'drug_{i}' for i in range(drug.shape[1])] +
        [f'clinical_{i}' for i in range(clinical.shape[1])] +
        ['tmb'] +
        [f'pathway_{i}' for i in range(20)] +
        list(interactions.keys()) +
        list(enrichment_scores.keys()) +
        ['clinical_risk_score'] +
        ['genomic_mean', 'genomic_std', 'genomic_max'] +
        ['drug_mean', 'drug_std']
    )
    
    # Normalize engineered features
    print("  Normalizing engineered features...")
    scaler = StandardScaler()
    X_engineered = scaler.fit_transform(X_engineered)
    
    # Metadata for later use
    metadata = {
        'pca': pca,
        'scaler': scaler,
        'feature_names': feature_names,
        'n_features': X_engineered.shape[1]
    }
    
    print(f"\nFeature engineering complete!")
    print(f"  Original features: {genomic.shape[1] + drug.shape[1] + clinical.shape[1]}")
    print(f"  Engineered features: {X_engineered.shape[1]}")
    print(f"  Samples: {X_engineered.shape[0]}")
    
    return X_engineered, feature_names, metadata


def main():
    """Main execution"""
    print("=" * 60)
    print("Advanced Feature Engineering for Cancer Genomics")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    X_genomic = np.load('artifacts/X_tabular_clean.npy')
    X_drug = np.load('artifacts/X_drug_fp_clean.npy')
    X_clinical = np.load('artifacts/X_clinical_aligned.npy')
    y = np.load('artifacts/y_clean.npy')
    
    # Load censoring info
    df = pd.read_csv('data/processed/ml_dataset_complete.csv')
    pfs_status = df['pfs_status'].values[:len(y)]
    
    print(f"  Genomic: {X_genomic.shape}")
    print(f"  Drug: {X_drug.shape}")
    print(f"  Clinical: {X_clinical.shape}")
    print(f"  Targets: {y.shape}")
    
    # Remove NaN rows
    mask = ~(np.isnan(X_genomic).any(axis=1) | 
             np.isnan(X_drug).any(axis=1) | 
             np.isnan(X_clinical).any(axis=1) | 
             np.isnan(y) |
             np.isnan(pfs_status))
    
    if not mask.all():
        print(f"  Removing {(~mask).sum()} rows with NaN values...")
        X_genomic = X_genomic[mask]
        X_drug = X_drug[mask]
        X_clinical = X_clinical[mask]
        y = y[mask]
        pfs_status = pfs_status[mask]
    
    # Engineer features
    X_engineered, feature_names, metadata = engineer_all_features(
        X_genomic, X_drug, X_clinical
    )
    
    # Save engineered features
    print("\nSaving engineered features...")
    np.save('artifacts/X_engineered.npy', X_engineered)
    np.save('artifacts/y_engineered.npy', y)
    np.save('artifacts/pfs_status_engineered.npy', pfs_status)
    
    with open('artifacts/feature_engineering_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"  Saved to artifacts/X_engineered.npy")
    print(f"  Shape: {X_engineered.shape}")
    print(f"  Range: [{X_engineered.min():.2f}, {X_engineered.max():.2f}]")
    
    # Print feature importance summary
    print("\nTop Engineered Features:")
    for i, name in enumerate(feature_names[-15:], start=len(feature_names)-15):
        print(f"  {i+1}. {name}")
    
    print("\n" + "=" * 60)
    print("Feature engineering complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
