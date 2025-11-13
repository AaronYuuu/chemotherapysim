"""
Download and prepare pathway data from KEGG and Reactome databases

Run this script once to download and cache pathway data

Author: Aaron Yu
Date: November 8, 2025
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.data.pathway_utils import PathwayGraphBuilder
import json


def main():
    """Download pathway data and build graph"""
    
    print("=" * 60)
    print("Pathway Data Download and Preparation")
    print("=" * 60)
    
    builder = PathwayGraphBuilder(cache_dir=str(PROJECT_ROOT / "data" / "pathway_graphs"))
    
    # Option 1: Try to download from KEGG
    print("\nAttempting to download KEGG pathways...")
    print("Note: This requires internet connection and may take several minutes")
    
    try:
        pathway_dict = builder.download_kegg_pathways(organism='hsa')
        
        if pathway_dict:
            print(f"\nSuccessfully downloaded {len(pathway_dict)} pathways")
            
            # Build graph
            print("\nBuilding pathway graph...")
            edge_index, gene_to_idx, idx_to_gene = builder.build_pathway_graph_from_dict(pathway_dict)
            
            # Save
            builder.save_graph('kegg_human_pathway_graph.pt')
            
            # Save pathway info separately
            pathway_info = {
                'num_pathways': len(pathway_dict),
                'num_genes': len(gene_to_idx),
                'num_edges': edge_index.shape[1],
                'pathway_ids': list(pathway_dict.keys())[:10]
            }
            
            info_path = PROJECT_ROOT / "data" / "pathway_graphs" / "pathway_info.json"
            with open(info_path, 'w') as f:
                json.dump(pathway_info, f, indent=2)
            
            print("\nPathway data preparation complete!")
            print(f"  Total pathways: {pathway_info['num_pathways']}")
            print(f"  Total genes: {pathway_info['num_genes']}")
            print(f"  Total edges: {pathway_info['num_edges']}")
            
        else:
            print("\nKEGG download failed or returned no data")
            raise Exception("KEGG download failed")
    
    except Exception as e:
        print(f"\nKEGG download failed: {e}")
        print("\nFalling back to creating synthetic test graph...")
        
        # Option 2: Create synthetic graph for testing
        edge_index = builder.create_simple_graph(num_genes=1318, density=0.01)
        builder.save_graph('test_pathway_graph.pt')
        
        print("\nCreated synthetic pathway graph for testing")
        print(f"  Nodes: {len(builder.gene_to_idx)}")
        print(f"  Edges: {edge_index.shape[1]}")
        print("\nNote: This is a random graph for testing only.")
        print("For real pathway data, ensure internet connection and try again.")
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
