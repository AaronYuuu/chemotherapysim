"""
Enhanced GNN Model with Clinical Variables and Multi-Headed Attention

Incorporates:
- Genomic features via GNN
- Drug fingerprints via MLP
- Clinical variables via MLP encoder
- Multi-headed cross-attention between all modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

try:
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False
    print("Warning: torch_geometric not installed. GNN will use fallback MLP mode.")


class ClinicalEncoder(nn.Module):
    """
    Encodes clinical variables into embeddings
    Clinical features are typically: age, stage, grade, treatment_line, etc.
    """
    def __init__(
        self,
        clinical_dim: int = 8,  # number of clinical features
        embed_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        self.clinical_dim = clinical_dim
        self.embed_dim = embed_dim
        
        # Clinical features are often small in number, so use moderate hidden layer
        self.encoder = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Clinical features [batch_size, clinical_dim]
        
        Returns:
            Clinical embeddings [batch_size, embed_dim]
        """
        return self.encoder(x)


class GenomicEncoder(nn.Module):
    """
    Encodes genomic features (gene expression) into embeddings
    Supports both flat features and graph-structured features
    """
    def __init__(
        self, 
        genomic_dim: int = 1318, 
        embed_dim: int = 256, 
        dropout: float = 0.4,
        use_graph: bool = False
    ):
        super().__init__()
        self.use_graph = use_graph and HAS_PYGEOMETRIC
        self.genomic_dim = genomic_dim
        self.embed_dim = embed_dim
        
        if self.use_graph:
            # Graph neural network layers
            self.conv1 = GATConv(1, 64, heads=4, dropout=dropout)
            self.conv2 = GATConv(64 * 4, 128, heads=4, dropout=dropout)
            self.conv3 = GCNConv(128 * 4, embed_dim)
        
        # Always create fallback MLP encoder
        self.mlp_encoder = nn.Sequential(
            nn.Linear(genomic_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Genomic features [batch_size, genomic_dim]
            edge_index: Graph edges [2, num_edges] (optional)
            batch: Not used in this implementation
        
        Returns:
            Genomic embeddings [batch_size, embed_dim]
        """
        if self.use_graph and edge_index is not None:
            # Process each sample in the batch through the graph
            batch_size = x.shape[0]
            embeddings = []
            
            for i in range(batch_size):
                # Reshape single sample: [genomic_dim] -> [genomic_dim, 1]
                x_sample = x[i].unsqueeze(1)  # [genomic_dim, 1]
                
                # GNN forward pass for this sample
                x_gnn = F.elu(self.conv1(x_sample, edge_index))
                x_gnn = F.elu(self.conv2(x_gnn, edge_index))
                x_gnn = self.conv3(x_gnn, edge_index)
                
                # Global pooling: [genomic_dim, embed_dim] -> [1, embed_dim]
                x_pooled = x_gnn.mean(dim=0, keepdim=True)
                embeddings.append(x_pooled)
            
            # Stack all embeddings: [batch_size, embed_dim]
            return torch.cat(embeddings, dim=0)
        else:
            # MLP forward pass
            return self.mlp_encoder(x)


class DrugEncoder(nn.Module):
    """Encodes drug fingerprints into embeddings"""
    def __init__(
        self, 
        drug_fp_dim: int = 8192, 
        embed_dim: int = 256, 
        dropout: float = 0.4
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(drug_fp_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Drug fingerprints [batch_size, drug_fp_dim]
        
        Returns:
            Drug embeddings [batch_size, embed_dim]
        """
        return self.encoder(x)


class TrimodalCrossAttention(nn.Module):
    """
    Multi-headed cross-attention mechanism between three modalities:
    - Genomic
    - Drug  
    - Clinical
    
    Each modality can attend to the others to capture interactions
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Genomic attends to Drug and Clinical
        self.genomic_to_drug = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.genomic_to_clinical = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Drug attends to Genomic and Clinical
        self.drug_to_genomic = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.drug_to_clinical = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Clinical attends to Genomic and Drug
        self.clinical_to_genomic = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.clinical_to_drug = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(
        self,
        genomic_embed: torch.Tensor,
        drug_embed: torch.Tensor,
        clinical_embed: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            genomic_embed: [batch_size, embed_dim]
            drug_embed: [batch_size, embed_dim]
            clinical_embed: [batch_size, embed_dim]
            return_attention: Whether to return attention weights
        
        Returns:
            fused_embed: [batch_size, embed_dim]
            attention_weights: Dictionary of attention weights (optional)
        """
        # Add sequence dimension for attention
        genomic = genomic_embed.unsqueeze(1)  # [batch, 1, embed_dim]
        drug = drug_embed.unsqueeze(1)
        clinical = clinical_embed.unsqueeze(1)
        
        # Cross-attention: genomic attends to drug and clinical
        genomic_drug, attn_gd = self.genomic_to_drug(genomic, drug, drug)
        genomic_clinical, attn_gc = self.genomic_to_clinical(genomic, clinical, clinical)
        
        # Cross-attention: drug attends to genomic and clinical
        drug_genomic, attn_dg = self.drug_to_genomic(drug, genomic, genomic)
        drug_clinical, attn_dc = self.drug_to_clinical(drug, clinical, clinical)
        
        # Cross-attention: clinical attends to genomic and drug
        clinical_genomic, attn_cg = self.clinical_to_genomic(clinical, genomic, genomic)
        clinical_drug, attn_cd = self.clinical_to_drug(clinical, drug, drug)
        
        # Aggregate attention outputs
        genomic_attended = genomic + genomic_drug + genomic_clinical
        drug_attended = drug + drug_genomic + drug_clinical
        clinical_attended = clinical + clinical_genomic + clinical_drug
        
        # Remove sequence dimension
        genomic_attended = genomic_attended.squeeze(1)
        drug_attended = drug_attended.squeeze(1)
        clinical_attended = clinical_attended.squeeze(1)
        
        # Concatenate and fuse
        combined = torch.cat([genomic_attended, drug_attended, clinical_attended], dim=-1)
        fused_embed = self.fusion(combined)
        
        attention_weights = None
        if return_attention:
            attention_weights = {
                'genomic_to_drug': attn_gd,
                'genomic_to_clinical': attn_gc,
                'drug_to_genomic': attn_dg,
                'drug_to_clinical': attn_dc,
                'clinical_to_genomic': attn_cg,
                'clinical_to_drug': attn_cd
            }
        
        return fused_embed, attention_weights


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head:
    - Primary: PFS prediction (regression)
    - Auxiliary: Resistance classification (binary)
    - Auxiliary: Pathway activity prediction
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_resistance_classes: int = 2,
        num_pathways: int = 50,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # PFS prediction head
        self.pfs_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # Resistance classification head
        self.resistance_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_resistance_classes)
        )
        
        # Pathway activity prediction head
        self.pathway_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_pathways)
        )
    
    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Args:
            x: Fused embeddings [batch_size, embed_dim]
            return_all: Whether to return all task predictions
        
        Returns:
            pfs_pred: PFS predictions [batch_size, 1]
            additional: Dictionary with auxiliary predictions (if return_all=True)
        """
        pfs_pred = self.pfs_head(x)
        
        if return_all:
            resistance_pred = self.resistance_head(x)
            pathway_pred = self.pathway_head(x)
            
            additional = {
                'resistance': resistance_pred,
                'pathway': pathway_pred
            }
            return pfs_pred, additional
        
        return pfs_pred


class PathwayAwareGNNWithClinical(nn.Module):
    """
    Complete model integrating:
    - Genomic features (via GNN or MLP)
    - Drug fingerprints (via MLP)
    - Clinical variables (via MLP)
    - Trimodal cross-attention
    - Multi-task learning
    """
    def __init__(
        self,
        genomic_dim: int = 1318,
        drug_fp_dim: int = 8192,
        clinical_dim: int = 8,
        embed_dim: int = 256,
        num_resistance_classes: int = 2,
        num_pathways: int = 50,
        use_pathway_graph: bool = False,
        dropout_genomic: float = 0.4,
        dropout_drug: float = 0.4,
        dropout_clinical: float = 0.3,
        dropout_head: float = 0.3,
        num_attention_heads: int = 8
    ):
        super().__init__()
        
        self.genomic_dim = genomic_dim
        self.drug_fp_dim = drug_fp_dim
        self.clinical_dim = clinical_dim
        self.embed_dim = embed_dim
        self.use_pathway_graph = use_pathway_graph
        
        # Encoders
        self.genomic_encoder = GenomicEncoder(
            genomic_dim, embed_dim, dropout_genomic, use_pathway_graph
        )
        self.drug_encoder = DrugEncoder(
            drug_fp_dim, embed_dim, dropout_drug
        )
        self.clinical_encoder = ClinicalEncoder(
            clinical_dim, embed_dim, dropout_clinical
        )
        
        # Trimodal cross-attention
        self.cross_attention = TrimodalCrossAttention(
            embed_dim, num_attention_heads, dropout=0.1
        )
        
        # Multi-task head
        self.multi_task_head = MultiTaskHead(
            embed_dim, num_resistance_classes, num_pathways, dropout_head
        )
    
    def forward(
        self,
        genomic_features: torch.Tensor,
        drug_features: torch.Tensor,
        clinical_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_all_tasks: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with three modalities
        
        Args:
            genomic_features: [batch_size, genomic_dim]
            drug_features: [batch_size, drug_fp_dim]
            clinical_features: [batch_size, clinical_dim]
            edge_index: Graph edges [2, num_edges] (optional)
            batch: Batch assignment (optional)
            return_attention: Whether to return attention weights
            return_all_tasks: Whether to return all task predictions
        
        Returns:
            pfs_prediction: Main PFS prediction [batch_size, 1]
            additional_outputs: Dictionary with attention weights and auxiliary predictions
        """
        # Encode features from each modality
        genomic_embed = self.genomic_encoder(genomic_features, edge_index, batch)
        drug_embed = self.drug_encoder(drug_features)
        clinical_embed = self.clinical_encoder(clinical_features)
        
        # Cross-attention between all modalities
        fused_embed, attention_weights = self.cross_attention(
            genomic_embed, drug_embed, clinical_embed, return_attention
        )
        
        # Multi-task predictions
        if return_all_tasks:
            pfs_pred, auxiliary_preds = self.multi_task_head(fused_embed, return_all=True)
            
            additional_outputs = {
                **auxiliary_preds,
                'embeddings': {
                    'genomic': genomic_embed,
                    'drug': drug_embed,
                    'clinical': clinical_embed,
                    'fused': fused_embed
                }
            }
            
            if attention_weights is not None:
                additional_outputs['attention'] = attention_weights
            
            return pfs_pred, additional_outputs
        else:
            pfs_pred = self.multi_task_head(fused_embed, return_all=False)
            
            additional_outputs = None
            if return_attention:
                additional_outputs = {'attention': attention_weights}
            
            return pfs_pred, additional_outputs if additional_outputs else pfs_pred
