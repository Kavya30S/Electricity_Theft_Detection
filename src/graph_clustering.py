import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms.community import greedy_modularity_communities

def build_graph(features, threshold=0.9):
    """Build a graph based on feature similarity."""
    similarity_matrix = cosine_similarity(features)
    G = nx.Graph()
    for i in range(len(features)):
        G.add_node(i)
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])
    return G

def detect_clusters(G):
    """Detect clusters using community detection."""
    communities = list(greedy_modularity_communities(G))
    cluster_labels = [-1] * len(G.nodes)
    for cluster_id, community in enumerate(communities):
        for node in community:
            cluster_labels[node] = cluster_id
    return cluster_labels