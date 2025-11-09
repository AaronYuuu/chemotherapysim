"""
Advanced GNN-based model with multi-task learning and attention mechanisms
Compatible with existing infrastructure, can be used with or without pathway graphs

Author: Aaron Yu
Date: November 8, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import json
from pathlib import Path

try:
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False
    print("Warning: torch_geometric not installed. GNN will use fallback MLP mode.")


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
        else:
            # Fallback MLP encoder
            self.encoder = nn.Sequential(
                nn.Linear(genomic_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, embed_dim),
                nn.BatchNorm1d(embed_dim)
            )
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Genomic features [batch_size, genomic_dim] or [num_nodes, 1] for graph
            edge_index: Graph edges [2, num_edges] (optional)
            batch: Batch assignment for graph pooling [num_nodes] (optional)
        
        Returns:
            Genomic embeddings [batch_size, embed_dim]
        """
        if self.use_graph and edge_index is not None:
            # GNN forward pass
            x = F.elu(self.conv1(x, edge_index))
            x = F.elu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
            
            # Pool to batch level if needed
            if batch is not None:
                x = global_mean_pool(x, batch)
            else:
                x = x.mean(dim=0, keepdim=True)
            
            return x
        else:
            # MLP forward pass
            return self.encoder(x)


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
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Drug fingerprints [batch_size, drug_fp_dim]
        
        Returns:
            Drug embeddings [batch_size, embed_dim]
        """
        return self.encoder(x)


class MultiTaskHead(nn.Module):
    """Multi-task prediction heads for various clinical and biological outcomes"""
    def __init__(
        self, 
        input_dim: int = 512, 
        num_resistance_classes: int = 5,
        num_pathways: int = 50,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Main task: PFS prediction
        self.pfs_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Auxiliary task 1: Resistance mechanism classification
        self.resistance_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_resistance_classes)
        )
        
        # Auxiliary task 2: Pathway activity prediction
        self.pathway_head = nn.Sequential(
            nn.Linear(input_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_pathways)
        )
    
    def forward(
        self, 
        combined_features: torch.Tensor,
        genomic_features: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            combined_features: Fused genomic + drug features [batch_size, input_dim]
            genomic_features: Just genomic features [batch_size, input_dim//2]
            return_all: Whether to return all auxiliary predictions
        
        Returns:
            Dictionary of predictions
        """
        outputs = {}
        
        # Main prediction
        outputs['pfs'] = self.pfs_head(combined_features)
        
        if return_all:
            outputs['resistance_mechanism'] = self.resistance_head(combined_features)
            outputs['pathway_activity'] = self.pathway_head(genomic_features)
        
        return outputs


class PathwayAwareGNN(nn.Module):
    """
    Advanced GNN model with pathway-aware architecture and multi-task learning
    
    Features:
    - Graph neural networks for pathway structure
    - Cross-attention between genomic and drug features
    - Multi-task learning (PFS, resistance, pathway activity)
    - Backward compatible with flat feature inputs
    """
    def __init__(
        self,
        genomic_dim: int = 1318,
        drug_fp_dim: int = 8192,
        embed_dim: int = 256,
        num_resistance_classes: int = 5,
        num_pathways: int = 50,
        use_pathway_graph: bool = True,
        dropout_genomic: float = 0.4,
        dropout_drug: float = 0.4,
        dropout_head: float = 0.3
    ):
        super().__init__()
        
        self.genomic_dim = genomic_dim
        self.drug_fp_dim = drug_fp_dim
        self.embed_dim = embed_dim
        self.use_pathway_graph = use_pathway_graph
        
        # Encoders
        self.genomic_encoder = GenomicEncoder(
            genomic_dim=genomic_dim,
            embed_dim=embed_dim,
            dropout=dropout_genomic,
            use_graph=use_pathway_graph
        )
        
        self.drug_encoder = DrugEncoder(
            drug_fp_dim=drug_fp_dim,
            embed_dim=embed_dim,
            dropout=dropout_drug
        )
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Multi-task prediction heads
        self.multitask_head = MultiTaskHead(
            input_dim=embed_dim * 2,
            num_resistance_classes=num_resistance_classes,
            num_pathways=num_pathways,
            dropout=dropout_head
        )
    
    def forward(
        self,
        genomic_features: torch.Tensor,
        drug_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_all_tasks: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with flexible inputs
        
        Args:
            genomic_features: [batch_size, genomic_dim] or [num_nodes, 1]
            drug_features: [batch_size, drug_fp_dim]
            edge_index: Graph edges [2, num_edges] (optional)
            batch: Batch assignment [num_nodes] (optional)
            return_attention: Whether to return attention weights
            return_all_tasks: Whether to return all task predictions
        
        Returns:
            pfs_prediction: Main PFS prediction [batch_size, 1]
            additional_outputs: Dictionary with attention weights and auxiliary predictions
        """
        # Encode features
        genomic_embed = self.genomic_encoder(genomic_features, edge_index, batch)
        drug_embed = self.drug_encoder(drug_features)
        
        # Cross-attention: How does drug interact with genomic features?
        attended_genomic, attention_weights = self.cross_attention(
            query=drug_embed.unsqueeze(1),
            key=genomic_embed.unsqueeze(1),
            value=genomic_embed.unsqueeze(1)
        )
        attended_genomic = attended_genomic.squeeze(1)
        
        # Combine features
        combined_features = torch.cat([attended_genomic, drug_embed], dim=1)
        
        # Multi-task predictions
        outputs = self.multitask_head(
            combined_features, 
            genomic_embed,
            return_all=return_all_tasks
        )
        
        # Prepare return values
        pfs_prediction = outputs['pfs']
        additional_outputs = None
        
        if return_attention or return_all_tasks:
            additional_outputs = {}
            
            if return_attention:
                additional_outputs['attention_weights'] = attention_weights
            
            if return_all_tasks:
                additional_outputs['resistance_mechanism'] = outputs.get('resistance_mechanism')
                additional_outputs['pathway_activity'] = outputs.get('pathway_activity')
        
        if additional_outputs:
            return pfs_prediction, additional_outputs
        
        return pfs_prediction
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = 'cpu'):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
        
        Returns:
            Loaded model in eval mode
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Default configuration
            config = {
                'genomic_dim': 1318,
                'drug_fp_dim': 8192,
                'embed_dim': 256,
                'num_resistance_classes': 5,
                'num_pathways': 50,
                'use_pathway_graph': True,
                'dropout_genomic': 0.4,
                'dropout_drug': 0.4,
                'dropout_head': 0.3
            }
        
        # Create model
        model = cls(**config)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model.to(device)
        
        return model
    
    def save_checkpoint(self, path: str, optimizer=None, epoch=None, metrics=None):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer state
            epoch: Current epoch
            metrics: Training metrics
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'genomic_dim': self.genomic_dim,
                'drug_fp_dim': self.drug_fp_dim,
                'embed_dim': self.embed_dim,
                'use_pathway_graph': self.use_pathway_graph
            }
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, path)
