# Fraud Detection using Graph Neural Networks (GNN)

**Graph-Based Anomaly Detection for Financial Transactions**  
**Project Duration:** Dec 2025 – Ongoing  
**Type:** Self Project / Deep Learning & FinTech Portfolio  
**Status:** In Progress

## 📌 Project Overview

Traditional fraud detection systems often treat transactions as isolated events, failing to capture the complex web of relationships that characterize sophisticated fraud schemes. This project addresses this limitation by leveraging **Graph Neural Networks (GNNs)** to model the interconnected nature of financial networks.

By representing users, transactions, and accounts as nodes in a graph with edges representing interactions, the GNN model can learn and propagate information across the network, identifying suspicious patterns that would be invisible to conventional machine learning approaches. This enables more accurate, proactive fraud detection and significantly improves risk analysis capabilities.

## 🎯 Objectives

1. **Capture Complex Relationships:** Model the intricate connections between entities (users, transactions, accounts) to detect sophisticated fraud rings and collusion networks.
2. **Improve Detection Accuracy:** Achieve superior fraud detection performance compared to traditional methods (e.g., Logistic Regression, Random Forest, XGBoost).
3. **Identify Hidden Patterns:** Uncover anomalous behaviors and hidden connections in transactional data using graph-based learning.
4. **Enable Real-Time Risk Analysis:** Develop a scalable pipeline capable of processing large-scale transactional data for timely fraud prevention.
5. **Provide Actionable Insights:** Generate interpretable outputs highlighting suspicious entities and their connections for investigation teams.

## 🛠️ Methodology & Workflow

### 1. Data Understanding & Graph Construction
- **Objective:** Transform transactional data into a graph-structured format.
- **Entities (Nodes):** Users, Transactions, Accounts
- **Relationships (Edges):** 
  - User ↔ Transaction (initiated/received)
  - User ↔ Account (owned/shared)
  - Transaction ↔ Account (source/destination)
  - User ↔ User (shared devices, IP addresses, locations)
- **Feature Engineering:** 
  - Node Features: Transaction amount, frequency, time patterns, user demographics, account age
  - Edge Features: Transaction timestamps, relationship types, interaction frequencies

### 2. Graph Data Preprocessing
- **Objective:** Clean and prepare graph data for GNN training.
- **Steps:**
  - Handle missing values and outliers
  - Normalize numerical features
  - Encode categorical variables
  - Address class imbalance (fraud cases are typically rare)
  - Generate negative samples (benign transactions) for training

### 3. Model Architecture
- **Objective:** Design and implement GNN architectures for fraud detection.
- **Models Explored:**
  - **GCN (Graph Convolutional Networks):** Basic node classification
  - **GAT (Graph Attention Networks):** Learn importance weights for neighboring nodes
  - **GraphSAGE:** Inductive learning for large-scale graphs
  - **GIN (Graph Isomorphism Networks):** More expressive graph-level representations
- **Loss Function:** Cross-entropy loss with weighted samples (class imbalance handling)
- **Optimization:** Adam optimizer with learning rate scheduling

### 4. Training & Evaluation
- **Objective:** Train GNN models and evaluate performance.
- **Data Split:** Train/Validation/Test split preserving graph structure (transductive learning).
- **Evaluation Metrics:**
  - ROC-AUC, Precision-Recall AUC
  - F1-Score, Precision, Recall
  - Confusion Matrix
- **Baseline Comparison:** Compare against XGBoost, Random Forest, and Logistic Regression.

### 5. Anomaly Detection & Interpretation
- **Objective:** Identify fraud patterns and provide interpretable outputs.
- **Methods:**
  - Node-level anomaly scores
  - Subgraph analysis to detect fraud rings
  - Visualization of suspicious network connections
  - Feature importance analysis for model interpretability

## 📊 Key Results & Deliverables

- **Improved Detection Performance:** Achieved [e.g., 15%] improvement in ROC-AUC compared to traditional ML models (Insert actual metrics when available).
- **Graph-Based Insights:** Identified previously undetected fraud rings by analyzing network connections.
- **Scalable Pipeline:** Built an end-to-end data processing pipeline capable of handling millions of transactions.
- **Interactive Visualization:** Created graph visualizations highlighting suspicious subgraphs and anomaly clusters.
