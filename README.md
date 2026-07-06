# Fraud-Detection-using-GNN

A sophisticated fraud detection system leveraging Graph Neural Networks to capture complex relationships between entities (users, transactions, accounts) for superior fraud pattern detection.

Features • Why GNN • Installation • Usage • Results • Project Structure
📊 Overview
This project implements a graph-based fraud detection system that models financial transactions as a graph where entities (users, accounts, merchants) are nodes and transactions are edges. Unlike traditional ML approaches that analyze transactions in isolation, this system identifies fraud by understanding network patterns and relationships between entities.

🎯 Key Capabilities
Feature	Description
Graph Construction	Build heterogeneous graphs from transactional data using PyTorch Geometric
GNN Models	GCN, GraphSAGE, and GAT architectures for node classification
Community Detection	Identify fraud rings and suspicious clusters using Louvain/Newman-Girvan
Real-time Detection	Process transactions with millisecond-level latency
Explainability	Subgraph highlighting and feature attribution for flagged transactions
Imbalanced Data Handling	Class weighting and sampling strategies for fraud detection
🎯 Fraud Patterns Detected
Fraud Rings - Groups of accounts working together to commit fraud

Account Takeover - Sudden behavioral changes indicating compromised accounts

Money Laundering - Circular transaction patterns and layering techniques

Synthetic Identity Fraud - Fabricated identities using shared attributes

Card Testing - Patterns where stolen cards are validated with small transactions


