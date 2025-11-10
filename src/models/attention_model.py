"""
Attention-Based Model for Cancer PFS Prediction

Uses multi-head self-attention to learn feature importance and interactions.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, roc_auc_score
from scipy.stats import spearmanr
import pickle


class FeatureAttention(nn.Module):
    """
    Self-attention over features to learn which features are most important.
    """
    def __init__(self, input_dim, num_heads=8, dropout=0.2):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [batch, features]
        # Reshape to [batch, 1, features] for attention
        x = x.unsqueeze(1)
        
        # Self-attention
        attended, attention_weights = self.attention(x, x, x)
        
        # Residual connection
        x = self.norm(x + self.dropout(attended))
        
        # Back to [batch, features]
        x = x.squeeze(1)
        
        return x, attention_weights


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward network.
    """
    def __init__(self, hidden_dim, num_heads=8, ff_dim=512, dropout=0.2):
        super().__init__()
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim]
        
        # Self-attention with residual
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class AttentionPFSPredictor(nn.Module):
    """
    Attention-based model for PFS prediction.
    
    Architecture:
    1. Feature embedding
    2. Feature-wise attention (learn which features matter)
    3. Transformer blocks (learn feature interactions)
    4. Global pooling
    5. Prediction head
    """
    def __init__(self, input_dim=244, hidden_dim=256, num_heads=8, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        
        # Feature embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Feature-wise attention
        self.feature_attention = FeatureAttention(
            hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=hidden_dim * 2,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Store attention weights for interpretability
        self.last_attention_weights = None
    
    def forward(self, x):
        # Embed features
        x = self.embedding(x)
        
        # Feature-wise attention
        x, attention_weights = self.feature_attention(x)
        self.last_attention_weights = attention_weights
        
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global pooling
        x = x.squeeze(1)  # [batch, hidden_dim]
        
        # Predict
        output = self.predictor(x)
        
        return output
    
    def get_feature_importance(self):
        """
        Get feature importance from attention weights.
        Returns normalized importance scores.
        """
        if self.last_attention_weights is None:
            return None
        
        # Average across heads and batch
        importance = self.last_attention_weights.mean(dim=(0, 1)).squeeze()
        
        # Normalize
        importance = importance / importance.sum()
        
        return importance.detach().cpu().numpy()


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_attention_model(X_train, y_train, X_val, y_val, 
                          input_dim=244, epochs=150, device='cpu',
                          hidden_dim=256, num_heads=8, num_layers=3,
                          dropout=0.3, learning_rate=0.001, batch_size=64,
                          verbose=True, save_path=None):
    """
    Train attention-based model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        input_dim: Input feature dimension
        epochs: Maximum training epochs
        device: Device to train on
        hidden_dim: Hidden layer dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        learning_rate: Learning rate
        batch_size: Batch size
        verbose: Print training progress
        save_path: Path to save best model (auto-generated if None)
    """
    if verbose:
        print("\nTraining Attention-Based Model...")
        print(f"  Input dimension: {input_dim}")
        print(f"  Device: {device}")
    
    # Auto-generate save path if not provided
    if save_path is None:
        import os
        import tempfile
        os.makedirs('artifacts/models', exist_ok=True)
        # Use a temporary unique name to avoid conflicts
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        save_path = f'artifacts/models/attention_{unique_id}_h{hidden_dim}_n{num_heads}_l{num_layers}.pth'
    
    # Create datasets
    train_dataset = SimpleDataset(X_train, y_train)
    val_dataset = SimpleDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = AttentionPFSPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25
    
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
                  f"Val Loss={val_loss:.4f}, MAE={val_mae:.4f}, "
                  f"Spearman={val_spearman:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model (only if file exists and was created during this training)
    import os
    if os.path.exists(save_path):
        try:
            model.load_state_dict(torch.load(save_path, weights_only=False))
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load saved model: {e}")
            # Continue with current model state
    
    if verbose:
        print(f"  Attention Model - Val MAE: {val_mae:.4f}, Spearman: {val_spearman:.4f}")
    
    return model


def predict_attention(model, X, device='cpu'):
    """Make predictions with attention model"""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy().flatten()
    
    return predictions


def get_feature_importance(model, X_sample, feature_names, device='cpu'):
    """
    Get feature importance from attention model.
    """
    model.eval()
    X_tensor = torch.FloatTensor(X_sample[:1]).to(device)
    
    with torch.no_grad():
        _ = model(X_tensor)
        importance = model.get_feature_importance()
    
    if importance is not None:
        # Get top features
        top_indices = np.argsort(importance)[-20:][::-1]
        
        print("\nTop 20 Most Important Features (from Attention):")
        for i, idx in enumerate(top_indices, 1):
            if idx < len(feature_names):
                print(f"  {i}. {feature_names[idx]}: {importance[idx]:.4f}")
    
    return importance


if __name__ == "__main__":
    # Test
    print("Testing Attention-Based Model...")
    
    batch_size = 32
    input_dim = 244
    
    X_dummy = torch.randn(batch_size, input_dim)
    y_dummy = torch.randn(batch_size, 1)
    
    model = AttentionPFSPredictor(input_dim=input_dim)
    output = model(X_dummy)
    
    print(f"Input shape: {X_dummy.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nAttention model test passed!")
