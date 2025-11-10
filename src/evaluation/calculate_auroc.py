"""
Calculate AUROC for GNN model by treating PFS prediction as classification
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.models.gnn_model import PathwayAwareGNN
from src.data.pathway_utils import PathwayGraphBuilder

# Paths
DATA_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_stratified" / "gnn_advanced"

class DrugResponseDataset(Dataset):
    """Dataset for drug response prediction"""
    def __init__(self, genomic_features, drug_features, targets, edge_index=None, use_graph=False):
        self.genomic_features = torch.FloatTensor(genomic_features)
        self.drug_features = torch.FloatTensor(drug_features)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)
        self.edge_index = edge_index
        self.use_graph = use_graph
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'genomic': self.genomic_features[idx],
            'drug': self.drug_features[idx],
            'target': self.targets[idx],
            'edge_index': self.edge_index
        }

def collate_fn(batch):
    """Custom collate function to handle edge_index properly"""
    genomic = torch.stack([item['genomic'] for item in batch])
    drug = torch.stack([item['drug'] for item in batch])
    target = torch.stack([item['target'] for item in batch])
    edge_index = batch[0]['edge_index']
    
    return {
        'genomic': genomic,
        'drug': drug,
        'target': target,
        'edge_index': edge_index
    }

def main():
    print("=" * 60)
    print("Calculating AUROC for GNN Model")
    print("=" * 60)
    
    # Load config
    with open(CHECKPOINT_DIR / 'config.json', 'r') as f:
        config = json.load(f)
    
    # Load data
    print("\nLoading data...")
    X_genomic = np.load(DATA_DIR / "X_tabular_clean.npy")
    X_drug = np.load(DATA_DIR / "X_drug_fp_clean.npy")
    y = np.load(DATA_DIR / "y_pfs_clean.npy")
    
    # Apply same preprocessing as training
    min_len = min(len(X_genomic), len(X_drug), len(y))
    X_genomic = X_genomic[:min_len]
    X_drug = X_drug[:min_len]
    y = y[:min_len]
    
    # Remove NaN rows
    nan_mask = np.isnan(X_genomic).any(axis=1) | np.isnan(X_drug).any(axis=1) | np.isnan(y)
    if nan_mask.any():
        print(f"  Removing {nan_mask.sum()} rows with NaN...")
        X_genomic = X_genomic[~nan_mask]
        X_drug = X_drug[~nan_mask]
        y = y[~nan_mask]
    
    # Impute remaining NaN
    for i in range(X_genomic.shape[1]):
        col = X_genomic[:, i]
        if np.isnan(col).any():
            col_mean = np.nanmean(col)
            X_genomic[:, i] = np.where(np.isnan(col), col_mean if not np.isnan(col_mean) else 0, col)
    
    # Normalize
    genomic_mean = X_genomic.mean(axis=0)
    genomic_std = X_genomic.std(axis=0)
    genomic_std[genomic_std == 0] = 1
    X_genomic = (X_genomic - genomic_mean) / genomic_std
    
    print(f"  Data shape: {X_genomic.shape[0]} samples")
    
    # Load pathway graph if needed
    edge_index = None
    if config.get('use_pathway_graph', False):
        print("\n Loading pathway graph...")
        pathway_graph_dir = PROJECT_ROOT / "data" / "pathway_graphs"
        pathway_graph_file = pathway_graph_dir / "kegg_human_pathway_graph.pt"
        
        if pathway_graph_file.exists():
            builder = PathwayGraphBuilder(cache_dir=str(pathway_graph_dir))
            edge_index, gene_to_idx, idx_to_gene = builder.load_graph("kegg_human_pathway_graph.pt")
            
            if len(gene_to_idx) != 1318:
                builder = PathwayGraphBuilder(cache_dir=str(pathway_graph_dir))
                edge_index = builder.create_simple_graph(num_genes=1318, density=0.01)
        else:
            builder = PathwayGraphBuilder(cache_dir=str(pathway_graph_dir))
            edge_index = builder.create_simple_graph(num_genes=1318, density=0.01)
        
        print(f"  Graph loaded: {edge_index.shape[1]} edges")
    
    # Create dataset
    dataset = DrugResponseDataset(
        X_genomic, X_drug, y,
        edge_index=edge_index,
        use_graph=config.get('use_pathway_graph', False)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn if config.get('use_pathway_graph', False) else None
    )
    
    # Load model
    print("\nLoading model...")
    model = PathwayAwareGNN(**{k: v for k, v in config.items() if k in [
        'genomic_dim', 'drug_fp_dim', 'embed_dim', 'num_resistance_classes',
        'num_pathways', 'use_pathway_graph', 'dropout_genomic', 'dropout_drug',
        'dropout_head'
    ]})
    
    checkpoint = torch.load(CHECKPOINT_DIR / 'best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            genomic = batch['genomic']
            drug = batch['drug']
            target = batch['target']
            edge_index = batch.get('edge_index')
            
            pred = model(genomic, drug, edge_index=edge_index)
            predictions.append(pred.numpy())
            targets.append(target.numpy())
    
    predictions = np.concatenate(predictions).squeeze()
    targets = np.concatenate(targets).squeeze()
    
    print(f"  Generated {len(predictions)} predictions")
    
    # Calculate classification metrics using median threshold
    median_pfs = np.median(targets)
    print(f"\n  PFS median threshold: {median_pfs:.2f}")
    
    # Binary classification: PFS > median = 1 (good prognosis), <= median = 0 (poor prognosis)
    y_true_binary = (targets > median_pfs).astype(int)
    y_pred_scores = predictions  # Use raw predictions as scores
    
    # Calculate AUROC
    try:
        auroc = roc_auc_score(y_true_binary, y_pred_scores)
        print(f"\n  AUROC: {auroc:.4f}")
        
        # Also calculate at different thresholds
        thresholds = [np.percentile(targets, p) for p in [25, 50, 75]]
        print("\n  AUROC at different PFS thresholds:")
        for threshold in thresholds:
            y_true = (targets > threshold).astype(int)
            auroc_thresh = roc_auc_score(y_true, y_pred_scores)
            print(f"    PFS > {threshold:.2f}: AUROC = {auroc_thresh:.4f}")
        
        # Save results
        results = {
            'auroc_median': float(auroc),
            'median_threshold': float(median_pfs),
            'auroc_25th_percentile': float(roc_auc_score((targets > thresholds[0]).astype(int), y_pred_scores)),
            'auroc_75th_percentile': float(roc_auc_score((targets > thresholds[2]).astype(int), y_pred_scores))
        }
        
        with open(CHECKPOINT_DIR / 'auroc_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n  Results saved to {CHECKPOINT_DIR / 'auroc_metrics.json'}")
        
    except Exception as e:
        print(f"\n  Error calculating AUROC: {e}")
    
    print("\n" + "=" * 60)
    print("AUROC Calculation Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
