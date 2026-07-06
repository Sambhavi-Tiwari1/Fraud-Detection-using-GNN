"""
Graph Attention Network (GAT) for fraud detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from typing import Optional

class FraudGAT(nn.Module):
    """
    Graph Attention Network for fraud detection with attention mechanism
    """
    
    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 out_channels: int = 2, num_layers: int = 2,
                 heads: int = 4, dropout: float = 0.3):
        """
        Initialize GAT model
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden units
            out_channels: Number of output classes
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super(FraudGAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # First GAT layer (multi-head)
        self.conv1 = GATConv(in_channels, hidden_channels, 
                            heads=heads, dropout=dropout)
        
        # Hidden GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, 
                       heads=1, dropout=dropout)
            )
        
        # Final layer (single head)
        self.conv_last = GATConv(hidden_channels * heads, out_channels,
                                heads=1, concat=False, dropout=dropout)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        
        # Residual connection
        self.residual = nn.Linear(in_channels, out_channels) \
                       if in_channels != out_channels else nn.Identity()
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            data: PyG Data object with x, edge_index, edge_weight
            
        Returns:
            Node-level predictions
        """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        
        # First layer
        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.conv_last(x, edge_index, edge_weight)
        
        return x
    
    def predict(self, data: Data, threshold: float = 0.5) -> torch.Tensor:
        """
        Get predictions with threshold
        
        Args:
            data: PyG Data object
            threshold: Classification threshold
            
        Returns:
            Binary predictions
        """
        logits = self.forward(data)
        probs = F.softmax(logits, dim=1)
        preds = (probs[:, 1] > threshold).long()
        return preds, probs
    
    def get_attention(self, data: Data) -> torch.Tensor:
        """
        Get attention weights for explainability
        
        Returns:
            Attention weights per edge
        """
        # This would require modifying GATConv to return attention weights
        # Simplified: return edge_weights as attention proxy
        return data.edge_weight if hasattr(data, 'edge_weight') else None

class EnhancedTemporalGAT(nn.Module):
    """
    Enhanced GAT with temporal attention and residual connections
    For capturing time-dependent fraud patterns [citation:4]
    """
    
    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 out_channels: int = 2, num_layers: int = 3,
                 heads: int = 4, dropout: float = 0.3,
                 use_residual: bool = True):
        super(EnhancedTemporalGAT, self).__init__()
        
        self.use_residual = use_residual
        self.num_layers = num_layers
        
        # Temporal encoding for transaction timestamps
        self.temporal_encoder = nn.Linear(1, in_channels)
        
        # GAT layers with residual connections
        self.conv1 = GATConv(in_channels, hidden_channels,
                            heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels,
                       heads=1, dropout=dropout)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.conv_last = GATConv(hidden_channels * heads, out_channels,
                                heads=1, concat=False, dropout=dropout)
        
        # Residual skip connection
        if use_residual:
            self.skip = nn.Linear(in_channels, out_channels)
    
    def forward(self, data: Data, timestamps: Optional[torch.Tensor] = None):
        x, edge_index = data.x, data.edge_index
        
        # Add temporal encoding if timestamps provided
        if timestamps is not None:
            temporal_encoding = self.temporal_encoder(timestamps)
            x = x + temporal_encoding
        
        # First layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Store for residual
        residual_input = x
        
        # Hidden layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = bn(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        # Final layer
        x = self.conv_last(x, edge_index)
        
        # Residual connection
        if self.use_residual:
            x = x + self.skip(residual_input)
        
        return x
