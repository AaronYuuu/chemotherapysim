"""
DeepSurv Model for Cancer Progression-Free Survival Prediction

Based on:
- Katzman et al., "DeepSurv: personalized treatment recommender system 
  using a Cox proportional hazards deep neural network" (2018)
- Specifically designed for survival analysis with censored data
- Uses Cox partial likelihood loss instead of MSE

Key advantages over regression:
1. Handles censored data properly
2. Produces relative risk scores
3. Better calibrated predictions
4. More interpretable for clinical use
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureExtractor(nn.Module):
    """
    Multi-modal feature extractor for genomic, drug, and clinical data.
    Produces a unified embedding for survival analysis.
    """
    def __init__(self, genomic_dim=1318, drug_dim=8192, clinical_dim=8, 
                 hidden_dim=256, dropout=0.3):
        super().__init__()
        
        # Genomic encoder (smaller network)
        self.genomic_encoder = nn.Sequential(
            nn.Linear(genomic_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Drug encoder (handles high-dimensional fingerprints)
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Clinical encoder (small network for 8 features)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention-based fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, genomic, drug, clinical):
        # Encode each modality
        genomic_embed = self.genomic_encoder(genomic)
        drug_embed = self.drug_encoder(drug)
        clinical_embed = self.clinical_encoder(clinical)
        
        # Stack for attention
        # Shape: [batch, 3, hidden_dim]
        stacked = torch.stack([genomic_embed, drug_embed, clinical_embed], dim=1)
        
        # Self-attention across modalities
        attended, _ = self.attention(stacked, stacked, stacked)
        
        # Concatenate attended representations
        # Shape: [batch, hidden_dim * 3]
        concatenated = attended.reshape(attended.size(0), -1)
        
        # Final fusion
        fused = self.fusion(concatenated)
        
        return fused


class DeepSurv(nn.Module):
    """
    DeepSurv model for survival analysis.
    
    Outputs a single risk score per patient. Higher risk = shorter survival.
    Uses Cox proportional hazards loss during training.
    """
    def __init__(self, genomic_dim=1318, drug_dim=8192, clinical_dim=8,
                 hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            genomic_dim=genomic_dim,
            drug_dim=drug_dim,
            clinical_dim=clinical_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # DeepSurv layers (risk predictor)
        layers = []
        input_dim = hidden_dim
        
        for i in range(num_layers):
            output_dim = hidden_dim // (2 ** i) if i < num_layers - 1 else hidden_dim // (2 ** (num_layers - 1))
            layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = output_dim
        
        self.risk_layers = nn.Sequential(*layers)
        
        # Final risk score (single output)
        self.risk_output = nn.Linear(input_dim, 1)
    
    def forward(self, genomic, drug, clinical):
        """
        Returns:
            risk_score: Shape [batch, 1]. Higher values = higher risk = shorter survival
        """
        # Extract features
        features = self.feature_extractor(genomic, drug, clinical)
        
        # Risk prediction
        risk_features = self.risk_layers(features)
        risk_score = self.risk_output(risk_features)
        
        return risk_score


class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards loss (negative partial log-likelihood).
    
    This is the proper loss function for survival analysis.
    Handles censored data naturally.
    
    Reference: Cox, D. R. (1972). Regression models and life-tables.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, risk_scores, times, events):
        """
        Args:
            risk_scores: Model output [batch, 1]
            times: Survival times [batch]
            events: Event indicators [batch] (1 = event occurred, 0 = censored)
        
        Returns:
            Negative partial log-likelihood
        """
        # Sort by time (descending)
        sorted_indices = torch.argsort(times, descending=True)
        risk_scores = risk_scores[sorted_indices].squeeze()
        events = events[sorted_indices]
        
        # Cox partial likelihood
        # For each event, compute risk compared to all patients at risk
        log_risk = risk_scores
        
        # Cumulative sum of exp(risk) for risk set
        # This is the denominator of the partial likelihood
        max_log_risk = log_risk.max()
        log_risk_normalized = log_risk - max_log_risk
        risk_set_sum = torch.cumsum(torch.exp(log_risk_normalized).flip(0), dim=0).flip(0)
        
        # Partial log-likelihood
        log_likelihood = log_risk - torch.log(risk_set_sum + 1e-7)
        
        # Only use uncensored events
        log_likelihood = log_likelihood * events
        
        # Average over events
        num_events = events.sum()
        if num_events > 0:
            loss = -log_likelihood.sum() / num_events
        else:
            loss = torch.tensor(0.0, device=risk_scores.device, requires_grad=True)
        
        return loss


class ConcordanceIndex:
    """
    C-index (concordance index) for survival analysis.
    
    Measures how well the model ranks patients by risk.
    C-index of 0.5 = random, 1.0 = perfect ranking.
    
    This is the standard metric for survival models (equivalent to AUROC).
    """
    def __call__(self, risk_scores, times, events):
        """
        Args:
            risk_scores: Predicted risk scores [N]
            times: Survival times [N]
            events: Event indicators [N]
        
        Returns:
            C-index value between 0 and 1
        """
        risk_scores = risk_scores.detach().cpu().numpy()
        times = times.detach().cpu().numpy() if times.requires_grad else times.cpu().numpy()
        events = events.detach().cpu().numpy() if events.requires_grad else events.cpu().numpy()
        
        n = len(risk_scores)
        concordant = 0
        discordant = 0
        tied_risk = 0
        
        # Compare all pairs
        for i in range(n):
            if events[i] == 0:  # Censored
                continue
            
            for j in range(n):
                if i == j:
                    continue
                
                # Patient i had event, patient j either:
                # 1. Had event later
                # 2. Was censored after patient i's event
                if times[j] > times[i]:
                    # Patient i (earlier event) should have higher risk
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += 1
                    else:
                        tied_risk += 1
        
        total_pairs = concordant + discordant + tied_risk
        
        if total_pairs == 0:
            return 0.5
        
        c_index = (concordant + 0.5 * tied_risk) / total_pairs
        return c_index


def create_event_indicator(pfs_values, median_pfs=None):
    """
    Create event indicator from PFS values.
    
    In real survival analysis, this comes from the data (event vs censored).
    For our regression setup, we'll treat:
    - Short PFS (< median) as event = 1
    - Long PFS (>= median) as event = 0 (censored)
    
    This allows us to use DeepSurv on regression data.
    """
    if median_pfs is None:
        median_pfs = np.median(pfs_values)
    
    events = (pfs_values < median_pfs).astype(np.float32)
    return events, median_pfs


if __name__ == "__main__":
    # Test the model
    print("Testing DeepSurv model...")
    
    batch_size = 32
    genomic_dim = 1318
    drug_dim = 8192
    clinical_dim = 8
    
    # Create dummy data
    genomic = torch.randn(batch_size, genomic_dim)
    drug = torch.randn(batch_size, drug_dim)
    clinical = torch.randn(batch_size, clinical_dim)
    times = torch.rand(batch_size) * 5  # PFS times
    events = torch.randint(0, 2, (batch_size,)).float()  # Event indicators
    
    # Create model
    model = DeepSurv(
        genomic_dim=genomic_dim,
        drug_dim=drug_dim,
        clinical_dim=clinical_dim,
        hidden_dim=256,
        num_layers=3,
        dropout=0.3
    )
    
    # Forward pass
    risk_scores = model(genomic, drug, clinical)
    print(f"Risk scores shape: {risk_scores.shape}")
    print(f"Risk scores range: [{risk_scores.min():.3f}, {risk_scores.max():.3f}]")
    
    # Test loss
    criterion = CoxPHLoss()
    loss = criterion(risk_scores, times, events)
    print(f"Cox loss: {loss.item():.4f}")
    
    # Test C-index
    c_index_fn = ConcordanceIndex()
    c_index = c_index_fn(risk_scores.squeeze(), times, events)
    print(f"C-index: {c_index:.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\nDeepSurv model test passed!")
