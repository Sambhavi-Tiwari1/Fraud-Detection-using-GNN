"""
Graph construction for fraud detection using PyTorch Geometric
"""
import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphBuilder:
    """
    Construct graph from tabular transaction data for GNN processing
    """
    
    def __init__(self, k_neighbors: int = 10, 
                 use_edge_weights: bool = True,
                 normalize: bool = True):
        """
        Initialize graph builder
        
        Args:
            k_neighbors: Number of neighbors for k-NN graph
            use_edge_weights: Whether to use transaction amounts as edge weights
            normalize: Whether to normalize node features
        """
        self.k_neighbors = k_neighbors
        self.use_edge_weights = use_edge_weights
        self.normalize = normalize
        
    def build_knn_graph(self, X: np.ndarray, 
                        y: np.ndarray = None,
                        amounts: np.ndarray = None) -> Data:
        """
        Build k-NN graph from transaction data
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=legitimate, 1=fraud)
            amounts: Transaction amounts for edge weights
            
        Returns:
            PyG Data object
        """
        logger.info(f"Building k-NN graph with k={self.k_neighbors}")
        
        n_samples = X.shape[0]
        
        # Normalize features
        if self.normalize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_norm = scaler.fit_transform(X)
        else:
            X_norm = X
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors, 
                                metric='euclidean')
        nbrs.fit(X_norm)
        distances, indices = nbrs.kneighbors(X_norm)
        
        # Build edge list (symmetrize)
        edge_index = []
        edge_weight = []
        
        for i in range(n_samples):
            for j in range(1, self.k_neighbors):  # Skip self (j=0)
                neighbor = indices[i, j]
                dist = distances[i, j]
                
                # Add undirected edge (symmetrize)
                edge_index.append([i, neighbor])
                edge_index.append([neighbor, i])
                
                if self.use_edge_weights:
                    # Weight by distance (closer = stronger)
                    weight = np.exp(-dist / distances[i, :].mean())
                    edge_weight.append(weight)
                    edge_weight.append(weight)
                else:
                    edge_weight.append(1.0)
                    edge_weight.append(1.0)
        
        # Convert to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float) if edge_weight else None
        
        # Node features
        x = torch.tensor(X_norm, dtype=torch.float)
        
        # Labels
        y_tensor = torch.tensor(y, dtype=torch.long) if y is not None else None
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y_tensor)
        
        logger.info(f"Graph built: {n_samples} nodes, {edge_index.shape[1]} edges")
        
        return data
    
    def build_heterogeneous_graph(self, transactions_df: pd.DataFrame) -> nx.Graph:
        """
        Build heterogeneous graph with users, transactions, and merchants
        
        Args:
            transactions_df: DataFrame with sender, receiver, amount, time, etc.
            
        Returns:
            NetworkX graph with multiple node types
        """
        G = nx.Graph()
        
        # Add nodes
        unique_users = pd.concat([
            transactions_df['sender_account'],
            transactions_df['receiver_account']
        ]).unique()
        
        for user in unique_users:
            G.add_node(user, type='user')
        
        # Add transaction nodes
        for idx, row in transactions_df.iterrows():
            tx_id = f"tx_{idx}"
            G.add_node(tx_id, type='transaction', 
                      amount=row['amount'],
                      time=row.get('time', 0))
            
            # Add edges
            G.add_edge(row['sender_account'], tx_id, type='sends')
            G.add_edge(tx_id, row['receiver_account'], type='received_by')
        
        logger.info(f"Heterogeneous graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def extract_graph_features(self, G: nx.Graph) -> pd.DataFrame:
        """
        Extract graph-based features for ML
        
        Features include:
        - Degree centrality
        - Clustering coefficient
        - PageRank
        - Betweenness centrality
        
        Returns:
            DataFrame with graph features
        """
        logger.info("Extracting graph features...")
        
        features = {}
        
        # Degree centrality
        degree = nx.degree_centrality(G)
        features['degree_centrality'] = degree
        
        # Clustering coefficient
        clustering = nx.clustering(G)
        features['clustering'] = clustering
        
        # PageRank
        pagerank = nx.pagerank(G)
        features['pagerank'] = pagerank
        
        # Eigenvector centrality (scaled)
        eigenvector = nx.eigenvector_centrality_numpy(G, max_iter=1000)
        features['eigenvector'] = eigenvector
        
        # Convert to DataFrame
        df = pd.DataFrame(features).fillna(0)
        
        logger.info(f"Extracted {df.shape[1]} graph features for {df.shape[0]} nodes")
        
        return df
    
    def detect_communities(self, G: nx.Graph, method: str = 'louvain'):
        """
        Detect communities in graph using Louvain algorithm
        
        Returns:
            Dictionary mapping node to community ID
        """
        logger.info(f"Detecting communities using {method}...")
        
        if method == 'louvain':
            try:
                import community.community_louvain as community_louvain
                partition = community_louvain.best_partition(G)
            except ImportError:
                logger.warning("python-louvain not installed. Using greedy modularity.")
                partition = nx.community.greedy_modularity_communities(G)
                # Convert to dict
                partition_dict = {}
                for i, comm in enumerate(partition):
                    for node in comm:
                        partition_dict[node] = i
                return partition_dict
        else:
            # Fallback
            partition = nx.community.greedy_modularity_communities(G)
            partition_dict = {}
            for i, comm in enumerate(partition):
                for node in comm:
                    partition_dict[node] = i
            return partition_dict
        
        return partition
