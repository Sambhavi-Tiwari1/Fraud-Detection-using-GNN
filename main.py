#!/usr/bin/env python
"""
Complete pipeline for GNN-based fraud detection
"""
import os
import sys
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph_builder import GraphBuilder
from src.models.gat import FraudGAT
from src.models.graphsage import FraudGraphSAGE
from src.models.gcn import FraudGCN
from src.training.trainer import GNNTrainer
from src.visualization.plots import Plotter
from src.utils.helpers import load_config, setup_logging, create_directories

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='GNN Fraud Detection Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/fraud_data.csv',
                       help='Path to data file')
    parser.add_argument('--train', action='store_true',
                       help='Train a new model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate existing model')
    parser.add_argument('--model_type', type=str, default='gat',
                       choices=['gcn', 'graphsage', 'gat'],
                       help='GNN architecture to use')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging()
    create_directories(['models', 'results/figures', 'data/processed'])
    
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("GNN FRAUD DETECTION SYSTEM")
    logger.info("="*60)
    
    # Step 1: Load and preprocess data
    logger.info("Loading data...")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} transactions")
    
    # Handle class imbalance
    fraud_ratio = df['Class'].mean()
    logger.info(f"Fraud ratio: {fraud_ratio:.2%}")
    
    # Step 2: Prepare features and labels
    feature_cols = [col for col in df.columns if col.startswith('V')] + ['Amount']
    X = df[feature_cols].values
    y = df['Class'].values
    
    # Step 3: Build graph
    logger.info("Building graph...")
    builder = GraphBuilder(k_neighbors=config['graph']['k_neighbors'],
                          normalize=True)
    
    data = builder.build_knn_graph(X, y)
    logger.info(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
    
    # Step 4: Train/val/test split
    n_nodes = data.num_nodes
    indices = np.arange(n_nodes)
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, 
                                          random_state=42, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5,
                                        random_state=42, stratify=y[temp_idx])
    
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    logger.info(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
    
    # Step 5: Initialize model
    logger.info(f"Initializing {args.model_type.upper()} model...")
    
    if args.model_type == 'gcn':
        from src.models.gcn import FraudGCN
        model = FraudGCN(data.num_features, 64, 2, num_layers=2)
    elif args.model_type == 'graphsage':
        model = FraudGraphSAGE(data.num_features, 64, 2, num_layers=2)
    else:  # gat
        model = FraudGAT(data.num_features, 64, 2, num_layers=2)
    
    logger.info(f"Model: {model.__class__.__name__} with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Step 6: Train
    if args.train:
        logger.info("Training model...")
        trainer = GNNTrainer(model, 
                            learning_rate=config['model']['learning_rate'],
                            class_weight=True)
        
        history = trainer.train(data, train_mask, val_mask,
                               epochs=config['training']['epochs'],
                               patience=config['training']['early_stopping_patience'])
        
        # Save model
        trainer.save_model('models/gnn_fraud_model.pt')
        
        # Plot training history
        plotter = Plotter()
        plotter.plot_training_history(history, 'results/figures/training_history.png')
    
    # Step 7: Evaluate
    if args.evaluate:
        logger.info("Evaluating model...")
        trainer = GNNTrainer(model)
        trainer.load_model('models/gnn_fraud_model.pt')
        
        metrics = trainer.test(data, test_mask)
        
        logger.info("\nTest Results:")
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                logger.info(f"  {key}: {value:.4f}")
        
        logger.info(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        for row in cm:
            logger.info(f"  {row}")
        
        # Plot results
        plotter = Plotter()
        plotter.plot_confusion_matrix(cm, 'results/figures/confusion_matrix.png')
        
        # Get predictions for explainability
        preds, probs = trainer.predict(data)
        
        # Detect fraud rings
        builder = GraphBuilder()
        G = builder.build_heterogeneous_graph(df)
        communities = builder.detect_communities(G, method='louvain')
        
        # Find suspicious communities
        suspicious_nodes = [i for i, p in enumerate(preds) if p == 1]
        suspicious_communities = {}
        for node in suspicious_nodes:
            comm = communities.get(node, -1)
            if comm not in suspicious_communities:
                suspicious_communities[comm] = []
            suspicious_communities[comm].append(node)
        
        logger.info(f"Detected {len(suspicious_communities)} suspicious communities")
        for comm, nodes in suspicious_communities.items():
            if len(nodes) >= 3:
                logger.info(f"  Community {comm}: {len(nodes)} suspicious nodes")

if __name__ == "__main__":
    main()
