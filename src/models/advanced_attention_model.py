"""
Advanced Multi-Modal Attention Model with Cross-Attention

Features:
1. Cross-attention between genomic, drug, and clinical features
2. Bidirectional information flow
3. Multi-head attention for different relationship perspectives
4. Feature-aware gating mechanisms
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MultiModalEmbedding(nn.Module):
    """Separate embeddings for genomic, drug, and clinical features"""
    def __init__(self, genomic_dim, drug_dim, clinical_dim, hidden_dim):
        super().__init__()
        
        self.genomic_embed = nn.Sequential(
            nn.Linear(genomic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.drug_embed = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.clinical_embed = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, genomic, drug, clinical):
        genomic_emb = self.genomic_embed(genomic)  # [batch, hidden]
        drug_emb = self.drug_embed(drug)          # [batch, hidden]
        clinical_emb = self.clinical_embed(clinical)  # [batch, hidden]
        
        return genomic_emb, drug_emb, clinical_emb


class CrossModalAttention(nn.Module):
    """Cross-attention between two modalities"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, query, key_value):
        """
        query: [batch, hidden] - the modality receiving information
        key_value: [batch, hidden] - the modality providing information
        """
        # Add sequence dimension
        query = query.unsqueeze(1)  # [batch, 1, hidden]
        key_value = key_value.unsqueeze(1)  # [batch, 1, hidden]
        
        # Cross-attention
        attn_out, attn_weights = self.cross_attention(query, key_value, key_value)
        query = self.norm1(query + attn_out)
        
        # Feed-forward
        ff_out = self.ff(query)
        query = self.norm2(query + ff_out)
        
        return query.squeeze(1), attn_weights


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional information exchange between modalities"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        # A -> B attention (A receives info from B)
        self.attn_a_from_b = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        # B -> A attention (B receives info from A)
        self.attn_b_from_a = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        # Gating mechanism to control information flow
        self.gate_a = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.gate_b = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, a, b):
        """
        Bidirectional information exchange between modalities a and b
        """
        # Get attended representations
        a_attended, _ = self.attn_a_from_b(a, b)  # A receiving from B
        b_attended, _ = self.attn_b_from_a(b, a)  # B receiving from A
        
        # Gating: decide how much to incorporate cross-modal info
        gate_a_val = self.gate_a(torch.cat([a, a_attended], dim=-1))
        gate_b_val = self.gate_b(torch.cat([b, b_attended], dim=-1))
        
        # Combine original and attended with gating
        a_out = gate_a_val * a + (1 - gate_a_val) * a_attended
        b_out = gate_b_val * b + (1 - gate_b_val) * b_attended
        
        return a_out, b_out


class TriModalCrossAttention(nn.Module):
    """Three-way cross-attention: Genomic ↔ Drug ↔ Clinical"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Pairwise bidirectional attention
        self.genomic_drug_attn = BidirectionalCrossAttention(hidden_dim, num_heads, dropout)
        self.genomic_clinical_attn = BidirectionalCrossAttention(hidden_dim, num_heads, dropout)
        self.drug_clinical_attn = BidirectionalCrossAttention(hidden_dim, num_heads, dropout)
        
        # Final normalization
        self.norm_genomic = nn.LayerNorm(hidden_dim)
        self.norm_drug = nn.LayerNorm(hidden_dim)
        self.norm_clinical = nn.LayerNorm(hidden_dim)
    
    def forward(self, genomic, drug, clinical):
        """
        Three-way information exchange
        """
        # Round 1: Genomic ↔ Drug
        genomic_1, drug_1 = self.genomic_drug_attn(genomic, drug)
        
        # Round 2: Genomic ↔ Clinical
        genomic_2, clinical_1 = self.genomic_clinical_attn(genomic_1, clinical)
        
        # Round 3: Drug ↔ Clinical
        drug_2, clinical_2 = self.drug_clinical_attn(drug_1, clinical_1)
        
        # Combine all information
        genomic_out = self.norm_genomic(genomic + genomic_1 + genomic_2)
        drug_out = self.norm_drug(drug + drug_1 + drug_2)
        clinical_out = self.norm_clinical(clinical + clinical_1 + clinical_2)
        
        return genomic_out, drug_out, clinical_out


class AdvancedMultiModalPredictor(nn.Module):
    """
    Advanced multi-modal predictor with cross-attention and bidirectional flow
    """
    def __init__(self, genomic_dim=100, drug_dim=100, clinical_dim=8, 
                 hidden_dim=256, num_heads=8, num_layers=3, dropout=0.3):
        super().__init__()
        
        self.genomic_dim = genomic_dim
        self.drug_dim = drug_dim
        self.clinical_dim = clinical_dim
        
        # Multi-modal embeddings
        self.embeddings = MultiModalEmbedding(
            genomic_dim, drug_dim, clinical_dim, hidden_dim
        )
        
        # Stack of cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            TriModalCrossAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Self-attention on combined representation
        self.self_attention = nn.MultiheadAttention(
            hidden_dim * 3, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_self = nn.LayerNorm(hidden_dim * 3)
        
        # Final prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """
        x: [batch, total_features]
        Splits into genomic, drug, clinical and processes with cross-attention
        """
        # Split features
        genomic = x[:, :self.genomic_dim]
        drug = x[:, self.genomic_dim:self.genomic_dim + self.drug_dim]
        clinical = x[:, self.genomic_dim + self.drug_dim:]
        
        # Embed each modality
        genomic_emb, drug_emb, clinical_emb = self.embeddings(genomic, drug, clinical)
        
        # Apply multiple layers of cross-attention
        for layer in self.cross_attention_layers:
            genomic_emb, drug_emb, clinical_emb = layer(genomic_emb, drug_emb, clinical_emb)
        
        # Concatenate all modalities
        combined = torch.cat([genomic_emb, drug_emb, clinical_emb], dim=-1)  # [batch, hidden*3]
        
        # Self-attention on combined representation
        combined_seq = combined.unsqueeze(1)  # [batch, 1, hidden*3]
        attn_out, _ = self.self_attention(combined_seq, combined_seq, combined_seq)
        combined = self.norm_self(combined_seq + attn_out).squeeze(1)
        
        # Final prediction
        output = self.predictor(combined)
        
        return output
    
    def get_attention_scores(self, x):
        """Extract attention scores for interpretability"""
        # This can be extended to return all attention weights
        pass


def train_advanced_attention_model(X_train, y_train, X_val, y_val,
                                   genomic_dim=100, drug_dim=100, clinical_dim=8,
                                   hidden_dim=256, num_heads=8, num_layers=3,
                                   dropout=0.3, learning_rate=0.001, batch_size=32,
                                   epochs=150, device='cpu', verbose=True):
    """
    Train advanced multi-modal attention model
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Training Advanced Multi-Modal Attention Model")
        print("=" * 60)
        print(f"  Architecture:")
        print(f"    - Genomic features: {genomic_dim}")
        print(f"    - Drug features: {drug_dim}")
        print(f"    - Clinical features: {clinical_dim}")
        print(f"    - Hidden dimension: {hidden_dim}")
        print(f"    - Attention heads: {num_heads}")
        print(f"    - Cross-attention layers: {num_layers}")
        print(f"    - Dropout: {dropout}")
        print(f"  Training:")
        print(f"    - Learning rate: {learning_rate}")
        print(f"    - Batch size: {batch_size}")
        print(f"    - Device: {device}")
    
    # Create datasets
    train_dataset = SimpleDataset(X_train, y_train)
    val_dataset = SimpleDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = AdvancedMultiModalPredictor(
        genomic_dim=genomic_dim,
        drug_dim=drug_dim,
        clinical_dim=clinical_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()
        
        val_mae = mean_absolute_error(all_targets, all_preds)
        val_spearman, _ = spearmanr(all_targets, all_preds)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        if verbose and (epoch + 1) % 15 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, MAE={val_mae:.4f}, Spearman={val_spearman:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'artifacts/models/advanced_attention_best.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    import os
    if os.path.exists('artifacts/models/advanced_attention_best.pth'):
        try:
            model.load_state_dict(torch.load('artifacts/models/advanced_attention_best.pth', weights_only=False))
        except:
            if verbose:
                print("  Warning: Could not load best model, using current state")
    
    if verbose:
        print(f"  Final - Val MAE: {val_mae:.4f}, Spearman: {val_spearman:.4f}")
    
    return model


def predict_advanced_attention(model, X, device='cpu'):
    """Make predictions with advanced attention model"""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy().flatten()
    
    return predictions


if __name__ == "__main__":
    # Test the model
    print("Testing Advanced Multi-Modal Attention Model...")
    
    # Create dummy data
    batch_size = 32
    genomic_dim = 100
    drug_dim = 100
    clinical_dim = 8
    total_dim = genomic_dim + drug_dim + clinical_dim
    
    X = np.random.randn(batch_size, total_dim).astype(np.float32)
    
    model = AdvancedMultiModalPredictor(
        genomic_dim=genomic_dim,
        drug_dim=drug_dim,
        clinical_dim=clinical_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=3
    )
    
    X_tensor = torch.FloatTensor(X)
    output = model(X_tensor)
    
    print(f"Input shape: {X_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ Advanced attention model test passed!")
