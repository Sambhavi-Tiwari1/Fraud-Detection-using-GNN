"""
Training loop for GNN fraud detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GNNTrainer:
    """
    Trainer for GNN fraud detection models
    """
    
    def __init__(self, model: nn.Module, 
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0005,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 class_weight: bool = True):
        """
        Initialize trainer
        
        Args:
            model: GNN model
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization
            device: Device to use ('cuda' or 'cpu')
            class_weight: Whether to weight classes for imbalance
        """
        self.model = model.to(device)
        self.device = device
        self.class_weight = class_weight
        
        # Optimizer
        self.optimizer = Adam(model.parameters(), 
                              lr=learning_rate, 
                              weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                          factor=0.5, patience=10,
                                          min_lr=1e-7)
        
        # Loss function (with class weights for imbalance)
        if class_weight:
            # We'll set weights during training
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
        
        logger.info(f"Trainer initialized on {device}")
    
    def train_epoch(self, data, train_mask, batch_size: int = None):
        """
        Train for one epoch
        
        Args:
            data: PyG Data object
            train_mask: Boolean mask for training nodes
            batch_size: Batch size for mini-batch training
        """
        self.model.train()
        
        if batch_size is None:
            # Full-batch training
            self.optimizer.zero_grad()
            
            out = self.model(data)
            loss = self.criterion(out[train_mask], data.y[train_mask])
            
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
        else:
            # Mini-batch training (for large graphs)
            # Simplified: sample nodes for each batch
            train_indices = torch.where(train_mask)[0].tolist()
            np.random.shuffle(train_indices)
            
            total_loss = 0
            n_batches = len(train_indices) // batch_size + 1
            
            for i in range(0, len(train_indices), batch_size):
                batch_idx = train_indices[i:i+batch_size]
                batch_mask = torch.zeros_like(train_mask, dtype=torch.bool)
                batch_mask[batch_idx] = True
                
                self.optimizer.zero_grad()
                
                out = self.model(data)
                loss = self.criterion(out[batch_mask], data.y[batch_mask])
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            return total_loss / n_batches
    
    def train(self, data, train_mask, val_mask, 
              epochs: int = 100, batch_size: int = None,
              patience: int = 20, verbose: bool = True):
        """
        Full training loop
        
        Args:
            data: PyG Data object
            train_mask: Training nodes mask
            val_mask: Validation nodes mask
            epochs: Number of epochs
            batch_size: Batch size for mini-batch training
            patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None
        
        # Compute class weights if enabled
        if self.class_weight:
            class_counts = torch.bincount(data.y[train_mask])
            class_weights = 1.0 / class_counts.float()
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        for epoch in tqdm(range(epochs), desc="Training", disable=not verbose):
            # Training
            train_loss = self.train_epoch(data, train_mask, batch_size)
            
            # Validation
            val_loss, val_acc, val_f1 = self.evaluate(data, val_mask)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d}/{epochs} | "
                           f"Train Loss: {train_loss:.4f} | "
                           f"Val Loss: {val_loss:.4f} | "
                           f"Val F1: {val_f1:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        logger.info(f"Training complete. Best Val F1: {best_val_f1:.4f}")
        
        return self.history
    
    def evaluate(self, data, mask):
        """
        Evaluate model on given mask
        
        Args:
            data: PyG Data object
            mask: Node mask
            
        Returns:
            (loss, accuracy, f1_score)
        """
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(data)
            pred = out.argmax(dim=1)
            
            # Loss
            loss = self.criterion(out[mask], data.y[mask]).item()
            
            # Metrics
            y_true = data.y[mask].cpu().numpy()
            y_pred = pred[mask].cpu().numpy()
            
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            
        return loss, accuracy, f1
    
    def test(self, data, test_mask):
        """
        Evaluate on test set with comprehensive metrics
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(data)
            pred = out.argmax(dim=1)
            probs = F.softmax(out, dim=1)
            
            y_true = data.y[test_mask].cpu().numpy()
            y_pred = pred[test_mask].cpu().numpy()
            y_probs = probs[test_mask, 1].cpu().numpy()
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_true, y_probs)
            }
            
            # Additional metrics for imbalanced data
            if len(np.unique(y_true)) == 2:
                # Binary classification - calculate fraud detection metrics
                fraud_mask = y_true == 1
                if fraud_mask.any():
                    metrics['fraud_recall'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
                    metrics['fraud_precision'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
                    metrics['fraud_f1'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
            
        logger.info(f"Test Results: Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def predict(self, data, threshold: float = 0.5):
        """
        Get predictions with probability
        """
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(data)
            probs = F.softmax(out, dim=1)
            preds = (probs[:, 1] > threshold).long()
            
        return preds.cpu().numpy(), probs.cpu().numpy()
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'config': {
                'model_class': self.model.__class__.__name__,
                'device': self.device
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        logger.info(f"Model loaded from {path}")
