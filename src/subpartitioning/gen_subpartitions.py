import os
import pickle
import pymetis

##############################
# HELPER FUNCTIONS
##############################

def build_adjacency(bipartite_dict, node_features_dict):
    """
    Convert a bipartite netlist dictionary into an adjacency list for PyMetis.
    The graph will have M+N nodes:
      - 0..M-1 are instance nodes
      - M..M+N-1 are net nodes (offset by num_instances)
    """
    instance_idx = bipartite_dict["instance_idx"]
    net_idx = bipartite_dict["net_idx"]
    edge_index = bipartite_dict["edge_index"]

    num_instances = node_features_dict['num_instances']
    num_nets = node_features_dict['num_nets']
    total_nodes = num_instances + num_nets

    adjacency = [[] for _ in range(total_nodes)]
    for col in range(edge_index.shape[1]):
        inst_id = instance_idx[col]
        net_id = net_idx[col] + num_instances  # offset net IDs
        adjacency[inst_id].append(net_id)
        adjacency[net_id].append(inst_id)
    return adjacency

def run_pymetis_partition(adjacency, ufactor, nparts):
    """
    Partition the graph using PyMetis with the given ufactor and number of partitions.
    Returns:
      edgecut (int): Total cut size.
      parts (list of int): Partition label for each node.
    """
    opt = pymetis.Options()
    opt.ufactor = ufactor
    edgecut, parts = pymetis.part_graph(nparts, adjacency=adjacency, options=opt)
    return edgecut, parts

def compute_partition_conductance(adjacency, parts):
    """
    Computes the conductance for each partition.
    Conductance for a partition S is defined as:
      φ(S) = cut(S, S̄) / min(vol(S), vol(S̄))
    where:
      - cut(S, S̄) is the number of edges connecting S to its complement,
      - vol(S) is the sum of degrees of nodes in S.
    Returns:
      conductance_dict: Mapping each partition label to its conductance.
      partitions: Mapping each partition label to the set of node indices in that partition.
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
    return conductance_dict, partitions

def extract_subgraph_with_mapping(adjacency, node_indices):
    """
    Extract the induced subgraph from the full graph (given by 'adjacency')
    for the nodes in 'node_indices'. Returns a tuple (sub_adj, mapping) where:
      - sub_adj is the adjacency list for the subgraph (nodes re-indexed from 0 to len(node_indices)-1),
      - mapping is a dictionary mapping the new index back to the original node index.
    """
    sorted_nodes = sorted(set(node_indices))
    mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(sorted_nodes)}
    reverse_mapping = {old_idx: new_idx for new_idx, old_idx in mapping.items()}
    sub_adj = [[] for _ in range(len(sorted_nodes))]
    for old_idx in sorted_nodes:
        new_idx = reverse_mapping[old_idx]
        sub_adj[new_idx] = [reverse_mapping[nb] for nb in adjacency[old_idx] if nb in reverse_mapping]
    return sub_adj, mapping

def compute_hierarchical_subpartitions(adjacency, ufactor=600, nparts=10):
    """
    Partitions the full graph using the specified ufactor and nparts,
    then partitions each top-level partition (from the full graph) using the same parameters.
    
    Returns:
      hierarchical_subpartitions: A dictionary mapping each subpartition label (e.g., "A1", "A2", ..., "J10")
                                   to the list of original node indices in that subpartition.
      hierarchical_conductance: A dictionary mapping each subpartition label to its computed conductance.
    """
    # Partition the full graph into nparts top-level partitions.
    _, parts_full = run_pymetis_partition(adjacency, ufactor, nparts)
    top_partitions = {}
    for node, part in enumerate(parts_full):
        top_partitions.setdefault(part, []).append(node)
    
    hierarchical_subpartitions = {}
    hierarchical_conductance = {}
    for top_part, node_list in top_partitions.items():
        top_label = chr(65 + top_part)  # 0 -> 'A', 1 -> 'B', etc.
        sub_adj, mapping = extract_subgraph_with_mapping(adjacency, node_list)
        _, sub_parts = run_pymetis_partition(sub_adj, ufactor, nparts)
        sub_cond_dict, sub_partition_nodes = compute_partition_conductance(sub_adj, sub_parts)
        for sub_part, new_indices in sub_partition_nodes.items():
            original_nodes = [mapping[new_idx] for new_idx in new_indices]
            hierarchical_key = f"{top_label}{sub_part + 1}"
            hierarchical_subpartitions[hierarchical_key] = original_nodes
            hierarchical_conductance[hierarchical_key] = sub_cond_dict[sub_part]
    return hierarchical_subpartitions, hierarchical_conductance

##############################
# MAIN SCRIPT
##############################

def main():
    # Define the parent folder containing all the superblue files.
    parent_folder = os.path.join('..', '..', 'data', 'superblue')
    
    # Loop through each subfolder in the parent folder.
    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path):
            # Check that the required pickle files exist in this folder.
            bipartite_fp = os.path.join(folder_path, "bipartite.pkl")
            node_features_fp = os.path.join(folder_path, "node_features.pkl")
            if os.path.exists(bipartite_fp) and os.path.exists(node_features_fp):
                print(f"Processing folder: {folder_path}")
                # Load the data.
                with open(bipartite_fp, 'rb') as f:
                    bipartite_dict = pickle.load(f)
                with open(node_features_fp, 'rb') as f:
                    node_features_dict = pickle.load(f)
                # Build the full graph's adjacency list.
                adjacency = build_adjacency(bipartite_dict, node_features_dict)
                # Compute the hierarchical subpartitions (expected to be 100 subpartitions).
                subpartitions_dict, _ = compute_hierarchical_subpartitions(adjacency, ufactor=600, nparts=10)
                # Save the subpartitions dictionary as 'subpartitions100.pkl' in the folder.
                output_fp = os.path.join(folder_path, "subpartitions100.pkl")
                with open(output_fp, 'wb') as f:
                    pickle.dump(subpartitions_dict, f)
                print(f"Saved subpartitions to {output_fp}")
            else:
                print(f"Skipping folder {folder_path}: Required pkl files not found.")

if __name__ == "__main__":
    main()