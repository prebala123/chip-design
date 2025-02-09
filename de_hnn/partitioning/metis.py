import pickle
import numpy as np
import pymetis
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def build_adjacency(bipartite_dict, node_features_dict):
    """
    Convert a bipartite netlist dictionary into adjacency lists for PyMetis.
    The graph will have M+N nodes:
      - 0..M-1 are instances
      - M..M+N-1 are nets (offset by M)

    Returns:
      adjacency (list of lists): adjacency[i] is the neighbors of node i.
    """
    instance_idx = bipartite_dict["instance_idx"]
    net_idx      = bipartite_dict["net_idx"]
    edge_index   = bipartite_dict["edge_index"]

    num_instances = node_features_dict['num_instances']  # M
    num_nets      = node_features_dict['num_nets']       # N
    total_nodes   = num_instances + num_nets

    # Initialize adjacency
    adjacency = [[] for _ in range(total_nodes)]

    # Build the bipartite edges in an undirected manner
    for col in range(edge_index.shape[1]):
        inst_id = instance_idx[col]
        net_id  = net_idx[col] + num_instances  # offset net IDs by M

        adjacency[inst_id].append(net_id)
        adjacency[net_id].append(inst_id)

    return adjacency, num_instances, num_nets

def run_pymetis_partition(adjacency, ufactor, nparts=2):
    """
    Partition the given adjacency list using PyMetis into nparts.

    Returns:
      edgecut (int): The total cut size.
      parts (list of int): A partition label for each node (0..nparts-1).
    """
    opt = pymetis.Options()
    if ufactor:
        opt.ufactor = ufactor

    edgecut, parts = pymetis.part_graph(nparts, adjacency=adjacency, options=opt)
    return edgecut, parts

def compute_partition_conductance(adjacency, parts):
    """
    Computes the conductance for each partition.
    
    For a partition S, conductance is defined as:
      φ(S) = (# edges from S to complement) / min(volume(S), volume(complement))
    where volume(S) = sum(degree(node) for node in S).
    
    Returns:
      conductance_dict: mapping from partition label to its conductance.
    """
    partitions = {}
    for node, part in enumerate(parts):
        partitions.setdefault(part, set()).add(node)
    
    degrees = [len(neighbors) for neighbors in adjacency]
    total_volume = sum(degrees)
    
    conductance_dict = {}
    for part, nodes in partitions.items():
        volume_S = sum(degrees[node] for node in nodes)
        volume_complement = total_volume - volume_S
        cut = 0
        for node in nodes:
            for neighbor in adjacency[node]:
                if neighbor not in nodes:
                    cut += 1
        min_volume = min(volume_S, volume_complement)
        conductance = cut / min_volume if min_volume > 0 else 0.0
        conductance_dict[part] = conductance
    return conductance_dict

def compute_heatmap_matrices(adjacency, ufactor_values=range(0, 901, 100), nparts_values=range(10, 101, 10)):
    """
    For a given graph (via its adjacency list), compute two matrices over a grid of parameter settings:
      - max_conductance_matrix[i, j] = maximum conductance (over partitions) when using ufactor=ufactor_values[i] and nparts=nparts_values[j].
      - avg_conductance_matrix[i, j] = average conductance (over partitions) for that parameter combination.
      
    Returns:
      max_conductance_matrix, avg_conductance_matrix (both as numpy arrays).
    """
    n_u = len(ufactor_values)
    n_p = len(nparts_values)
    max_matrix = np.zeros((n_u, n_p))
    avg_matrix = np.zeros((n_u, n_p))
    
    for i, ufactor in enumerate(ufactor_values):
        for j, nparts in enumerate(nparts_values):
            _, parts = run_pymetis_partition(adjacency, ufactor, nparts)
            conductance_dict = compute_partition_conductance(adjacency, parts)
            max_matrix[i, j] = max(conductance_dict.values())
            avg_matrix[i, j] = sum(conductance_dict.values()) / len(conductance_dict)
    return max_matrix, avg_matrix

def extract_subgraph(adjacency, node_indices):
    """
    Extract the induced subgraph from the full graph given by 'adjacency' for the nodes in 'node_indices'.
    Returns a new adjacency list for the subgraph with nodes re-indexed from 0 to len(node_indices)-1.
    """
    node_set = set(node_indices)
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(node_set))}
    sub_adj = [[] for _ in range(len(node_set))]
    
    for old_idx in node_set:
        new_idx = mapping[old_idx]
        # Include only neighbors that are in the subgraph.
        sub_adj[new_idx] = [mapping[nb] for nb in adjacency[old_idx] if nb in node_set]
    return sub_adj

#########################
# MAIN ROUTINE
#########################

def main():
    # --- Load the full bipartite graph data ---
    bipartite_fp = os.path.join('..', 'superblue', 'superblue_18', 'bipartite.pkl')
    node_features_fp = os.path.join('..', 'superblue', 'superblue_18', 'node_features.pkl')
    
    with open(bipartite_fp, "rb") as f:
        bipartite_dict = pickle.load(f)
    with open(node_features_fp, "rb") as f:
        node_features_dict = pickle.load(f)
    
    full_adjacency, num_instances, num_nets = build_adjacency(bipartite_dict, node_features_dict)
    
    # --- Optimal partitioning on the full graph ---
    # You found that ufactor=600 and nparts=10 is optimal.
    optimal_ufactor = 600
    optimal_nparts = 10
    _, full_parts = run_pymetis_partition(full_adjacency, optimal_ufactor, optimal_nparts)
    
    # Organize nodes by partition (0 through 9)
    partitions = {}
    for node, part in enumerate(full_parts):
        partitions.setdefault(part, []).append(node)
    
    print("Full graph partitioning complete. Processing each of the 10 partitions...")

    # Prepare lists to store heatmap matrices for each subgraph.
    subgraph_max_matrices = []
    subgraph_avg_matrices = []
    
    # Define the parameter grid (same as before)
    ufactor_values = list(range(0, 901, 100))
    nparts_values = list(range(10, 101, 10))
    
    # For each partition (i.e., for each set of nodes in the full graph):
    for part in sorted(partitions.keys()):
        nodes = partitions[part]
        print(f"Processing subgraph for partition {part} with {len(nodes)} nodes...")
        # Extract the subgraph adjacency (re-index nodes)
        sub_adj = extract_subgraph(full_adjacency, nodes)
        # Compute heatmap matrices for this subgraph.
        max_matrix, avg_matrix = compute_heatmap_matrices(sub_adj, ufactor_values, nparts_values)
        subgraph_max_matrices.append(max_matrix)
        subgraph_avg_matrices.append(avg_matrix)
    
    #############################
    # Plotting: Arrange the 10 subgraph heatmaps in a 5x2 grid.
    #############################
    
    # --- Plot maximum conductance heatmaps ---
    fig_max, axes_max = plt.subplots(5, 2, figsize=(16, 20))
    axes_max = axes_max.flatten()
    for idx, max_matrix in enumerate(subgraph_max_matrices):
        ax = axes_max[idx]
        sns.heatmap(max_matrix, annot=True, fmt=".4f", cmap="viridis",
                    xticklabels=nparts_values, yticklabels=ufactor_values, ax=ax)
        ax.set_xlabel("Number of Partitions")
        ax.set_ylabel("Ufactor")
        ax.set_title(f"Partition {idx}: Maximum Conductance")
    plt.tight_layout()
    plt.savefig("subgraph_max_conductance_heatmaps.png", dpi=300)
    print("Maximum conductance heatmaps saved as 'subgraph_max_conductance_heatmaps.png'.")

    # --- Plot average conductance heatmaps ---
    fig_avg, axes_avg = plt.subplots(5, 2, figsize=(16, 20))
    axes_avg = axes_avg.flatten()
    for idx, avg_matrix in enumerate(subgraph_avg_matrices):
        ax = axes_avg[idx]
        sns.heatmap(avg_matrix, annot=True, fmt=".4f", cmap="viridis",
                    xticklabels=nparts_values, yticklabels=ufactor_values, ax=ax)
        ax.set_xlabel("Number of Partitions")
        ax.set_ylabel("Ufactor")
        ax.set_title(f"Partition {idx}: Average Conductance")
    plt.tight_layout()
    plt.savefig("subgraph_avg_conductance_heatmaps.png", dpi=300)
    print("Average conductance heatmaps saved as 'subgraph_avg_conductance_heatmaps.png'.")

if __name__ == "__main__":
    main()