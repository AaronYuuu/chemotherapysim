"""
Utilities for building and managing biological pathway graphs
Supports KEGG and Reactome pathway databases

Author: Aaron Yu
Date: November 8, 2025
"""

import json
import requests
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import time

try:
    from torch_geometric.data import Data
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False
    print("Warning: torch_geometric not installed")


class PathwayGraphBuilder:
    """
    Builds gene-gene interaction graphs from pathway databases
    """
    def __init__(self, cache_dir: str = "data/pathway_graphs"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.gene_to_idx = {}
        self.idx_to_gene = {}
        self.pathway_genes = {}
        self.edge_index = None
    
    def download_kegg_pathways(self, organism: str = 'hsa') -> Dict[str, List[str]]:
        """
        Download pathway information from KEGG database
        
        Args:
            organism: KEGG organism code (default: 'hsa' for human)
        
        Returns:
            Dictionary mapping pathway IDs to gene lists
        """
        cache_file = self.cache_dir / f"kegg_{organism}_pathways.json"
        
        if cache_file.exists():
            print(f"Loading cached KEGG pathways from {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        print(f"Downloading KEGG pathways for organism: {organism}")
        pathway_dict = {}
        
        try:
            # Get list of pathways
            url = f"http://rest.kegg.jp/list/pathway/{organism}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            pathways = []
            for line in response.text.strip().split('\n'):
                pathway_id = line.split('\t')[0]
                pathways.append(pathway_id)
            
            print(f"Found {len(pathways)} pathways")
            
            # Get genes for each pathway
            for i, pathway_id in enumerate(pathways):
                if i % 10 == 0:
                    print(f"Processing pathway {i+1}/{len(pathways)}")
                
                try:
                    url = f"http://rest.kegg.jp/get/{pathway_id}"
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    genes = []
                    in_gene_section = False
                    
                    for line in response.text.split('\n'):
                        if line.startswith('GENE'):
                            in_gene_section = True
                            gene_info = line.split()[1]
                            genes.append(gene_info)
                        elif in_gene_section:
                            if line.startswith(' '):
                                gene_info = line.strip().split()[0]
                                genes.append(gene_info)
                            else:
                                break
                    
                    if genes:
                        pathway_dict[pathway_id] = genes
                    
                    time.sleep(0.1)
                
                except Exception as e:
                    print(f"Error downloading pathway {pathway_id}: {e}")
                    continue
            
            # Cache results
            with open(cache_file, 'w') as f:
                json.dump(pathway_dict, f, indent=2)
            
            print(f"Downloaded {len(pathway_dict)} pathways with gene information")
            return pathway_dict
        
        except Exception as e:
            print(f"Error downloading KEGG pathways: {e}")
            return {}
    
    def build_pathway_graph_from_dict(
        self, 
        pathway_dict: Dict[str, List[str]],
        gene_list: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str]]:
        """
        Build gene interaction graph from pathway dictionary
        Genes in same pathway are connected
        
        Args:
            pathway_dict: Dictionary mapping pathway IDs to gene lists
            gene_list: Optional list of genes to include (filters graph)
        
        Returns:
            edge_index: Graph edges [2, num_edges]
            gene_to_idx: Mapping from gene name to node index
            idx_to_gene: Mapping from node index to gene name
        """
        # Collect all unique genes
        all_genes = set()
        for genes in pathway_dict.values():
            all_genes.update(genes)
        
        # Filter to gene_list if provided
        if gene_list is not None:
            gene_list_set = set(gene_list)
            all_genes = all_genes.intersection(gene_list_set)
        
        # Create gene index mapping
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(sorted(all_genes))}
        self.idx_to_gene = {idx: gene for gene, idx in self.gene_to_idx.items()}
        
        # Build edges: connect genes in same pathway
        edges = set()
        
        for pathway_id, genes in pathway_dict.items():
            pathway_genes_filtered = [g for g in genes if g in self.gene_to_idx]
            
            # Store pathway membership
            self.pathway_genes[pathway_id] = pathway_genes_filtered
            
            # Create edges between all genes in pathway
            for i, gene1 in enumerate(pathway_genes_filtered):
                for gene2 in pathway_genes_filtered[i+1:]:
                    idx1 = self.gene_to_idx[gene1]
                    idx2 = self.gene_to_idx[gene2]
                    edges.add((min(idx1, idx2), max(idx1, idx2)))
        
        # Convert to edge_index format
        if edges:
            edges_list = list(edges)
            edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
            
            # Make undirected (add reverse edges)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        self.edge_index = edge_index
        
        print(f"Built graph with {len(self.gene_to_idx)} nodes and {edge_index.shape[1]} edges")
        
        return edge_index, self.gene_to_idx, self.idx_to_gene
    
    def save_graph(self, filename: str = "pathway_graph.pt"):
        """Save graph and mappings to file"""
        save_path = self.cache_dir / filename
        
        data = {
            'edge_index': self.edge_index,
            'gene_to_idx': self.gene_to_idx,
            'idx_to_gene': self.idx_to_gene,
            'pathway_genes': self.pathway_genes
        }
        
        torch.save(data, save_path)
        print(f"Saved graph to {save_path}")
    
    def load_graph(self, filename: str = "pathway_graph.pt"):
        """Load graph and mappings from file"""
        load_path = self.cache_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Graph file not found: {load_path}")
        
        data = torch.load(load_path, weights_only=False)
        
        self.edge_index = data['edge_index']
        self.gene_to_idx = data['gene_to_idx']
        self.idx_to_gene = data['idx_to_gene']
        self.pathway_genes = data.get('pathway_genes', {})
        
        print(f"Loaded graph from {load_path}")
        print(f"  Nodes: {len(self.gene_to_idx)}")
        print(f"  Edges: {self.edge_index.shape[1]}")
        
        return self.edge_index, self.gene_to_idx, self.idx_to_gene
    
    def create_simple_graph(self, num_genes: int = 1318, density: float = 0.01) -> torch.Tensor:
        """
        Create a simple random graph for testing
        
        Args:
            num_genes: Number of gene nodes
            density: Edge density (fraction of possible edges)
        
        Returns:
            edge_index: Random graph edges
        """
        num_possible_edges = num_genes * (num_genes - 1) // 2
        num_edges = int(num_possible_edges * density)
        
        edges = set()
        while len(edges) < num_edges:
            i = np.random.randint(0, num_genes)
            j = np.random.randint(0, num_genes)
            if i != j:
                edges.add((min(i, j), max(i, j)))
        
        edges_list = list(edges)
        edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
        
        # Make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        self.edge_index = edge_index
        self.gene_to_idx = {f"gene_{i}": i for i in range(num_genes)}
        self.idx_to_gene = {i: f"gene_{i}" for i in range(num_genes)}
        
        print(f"Created random graph with {num_genes} nodes and {edge_index.shape[1]} edges")
        
        return edge_index


def prepare_genomic_data_for_gnn(
    genomic_features: np.ndarray,
    edge_index: torch.Tensor,
    gene_to_idx: Dict[str, int],
    feature_names: Optional[List[str]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert flat genomic features to graph format
    
    Args:
        genomic_features: [batch_size, num_features] numpy array
        edge_index: Graph structure [2, num_edges]
        gene_to_idx: Mapping from gene names to indices
        feature_names: Optional list of feature names
    
    Returns:
        node_features: [num_nodes, 1] node features
        batch: [num_nodes] batch assignment
    """
    batch_size = genomic_features.shape[0]
    num_nodes = len(gene_to_idx)
    
    # Convert to tensor
    genomic_tensor = torch.tensor(genomic_features, dtype=torch.float32)
    
    # Reshape for graph: [batch_size * num_nodes, 1]
    node_features = genomic_tensor.reshape(-1, 1)
    
    # Create batch assignment
    batch = torch.arange(batch_size).repeat_interleave(num_nodes)
    
    return node_features, batch


def get_pathway_masks(
    pathway_genes: Dict[str, List[str]],
    gene_to_idx: Dict[str, int],
    top_k: int = 10
) -> Dict[str, torch.Tensor]:
    """
    Create boolean masks for major pathways
    
    Args:
        pathway_genes: Dictionary mapping pathway IDs to gene lists
        gene_to_idx: Gene name to index mapping
        top_k: Number of top pathways to include
    
    Returns:
        Dictionary mapping pathway names to boolean masks
    """
    # Sort pathways by size
    sorted_pathways = sorted(
        pathway_genes.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )[:top_k]
    
    num_genes = len(gene_to_idx)
    pathway_masks = {}
    
    for pathway_id, genes in sorted_pathways:
        mask = torch.zeros(num_genes, dtype=torch.bool)
        
        for gene in genes:
            if gene in gene_to_idx:
                idx = gene_to_idx[gene]
                mask[idx] = True
        
        pathway_masks[pathway_id] = mask
    
    return pathway_masks


if __name__ == "__main__":
    # Example usage
    print("Testing PathwayGraphBuilder...")
    
    builder = PathwayGraphBuilder()
    
    # Option 1: Use existing KEGG graph (RECOMMENDED - already downloaded)
    print("\nLoading existing KEGG pathway graph...")
    edge_index, gene_to_idx, idx_to_gene = builder.load_graph('kegg_human_pathway_graph.pt')
    
    # Re-download from KEGG (only if  need to update)
    # pathway_dict = builder.download_kegg_pathways('hsa')
    # edge_index, gene_to_idx, idx_to_gene = builder.build_pathway_graph_from_dict(pathway_dict)
    # builder.save_graph('kegg_human_pathway_graph.pt')
   
    
    print("\nDone!")
