
import pickle
import numpy as np
import kahypar as kahypar
import os

def build_hypergraph(bipartite_dict, node_features_dict):
    """
    Convert the bipartite netlist dictionary into a hypergraph representation.

    - Hypergraph vertices = instances (0..M-1)
    - Hyperedges = nets, each net contains the instances it connects.
    """
    instance_idx = bipartite_dict["instance_idx"]   # [0..M-1]
    net_idx      = bipartite_dict["net_idx"]        # [0..N-1]
    edge_index   = bipartite_dict["edge_index"]     # shape (2, E)

    M = node_features_dict['num_instances']
    N = node_features_dict['num_nets']

    # Build net -> list of instances
    # net_membership[net_id] = [inst1, inst2, ...] for that net
    net_membership = [[] for _ in range(N)]
    for e in range(edge_index.shape[1]):
        inst_id = instance_idx[e]
        net_id  = net_idx[e]
        net_membership[net_id].append(inst_id)

    # Return the adjacency in "hyperedge" form
    # We'll have M vertices, N hyperedges
    return M, net_membership

def create_kahypar_hypergraph(num_vertices, net_membership):
    """
    Create an unweighted KaHyPar hypergraph object.

    :param num_vertices: number of vertices (instances)
    :param net_membership: list of lists, each sublist is the set of instance-IDs in that net
    :return: A kahypar.Hypergraph object
    """
    # For unweighted partitioning:
    vertex_weights = [1] * num_vertices
    edge_weights   = [1] * len(net_membership)

    # KaHyPar expects edges as a list of lists of vertex IDs.
    # The hypergraph is undirected by default.
    hypergraph = kahypar.Hypergraph(
        num_vertices=num_vertices,
        num_hyperedges=len(net_membership),
        # flatten each net's instance list?
        # actually, we can pass net_membership directly if it's a list of lists
        hyperedges=net_membership,
        edge_weights=edge_weights,
        vertex_weights=vertex_weights,
        directed=False
    )
    return hypergraph

def run_kahypar_partition(hypergraph, k=2, epsilon=0.03):
    """
    Partition the hypergraph into k parts, allowing up to epsilon imbalance.
    Returns the partition array: partition[node] = part_id.
    """
    # Initialize context
    # You can load presets like "default", "quality", "deterministic", or "balanced".
    # We'll do a minimal example with "default".
    context = kahypar.Context()
    context.loadPreset("default")

    # Number of parts
    context.setK(k)
    # Imbalance tolerance (3% in this example)
    context.setEpsilon(epsilon)
    # Objective: we want to minimize the hyperedge cut
    context.setObjective(kahypar.Objective.CUT)

    # Optionally suppress KaHyPar output in the console
    context.suppressOutput(True)

    # Partition the hypergraph
    partition = kahypar.partition(hypergraph, context)

    # Compute the cut size
    cut = hypergraph.cut(partition)
    return partition, cut

def store_partition_results_as_pkl(partition, filename="hypergraph_partition.pkl"):
    """
    Store the partition labels in a dictionary, then pickle it.
    Dictionary format: {vertex_id: partition_label}
    """
    partition_dict = {v: part_id for v, part_id in enumerate(partition)}

    with open(filename, "wb") as f:
        pickle.dump(partition_dict, f)
    print(f"Saved partition results to '{filename}'")

def main():
    # 1) Load bipartite dictionary from star.pkl
    bipartite_fp = os.path.join('..', 'superblue', 'superblue_18', 'bipartite.pkl')
    with open(bipartite_fp, "rb") as f:
        bipartite_dict = pickle.load(f)

    node_features_fp = os.path.join('..', 'superblue', 'superblue_18', 'node_features.pkl')
    with open(node_features_fp, "rb") as f:
        node_features_dict = pickle.load(f)

    # 2) Convert to hypergraph: M vertices, net_membership for edges
    M, net_membership = build_hypergraph(bipartite_dict, node_features_dict)

    # 3) Create a KaHyPar Hypergraph (unweighted)
    hypergraph = create_kahypar_hypergraph(M, net_membership)

    # 4) Partition the hypergraph into 2 parts (k=2), 3% imbalance
    partition, cut = run_kahypar_partition(hypergraph, k=2, epsilon=0.03)

    print("=== KaHyPar Hypergraph Partitioning ===")
    print(f"Hyperedge cut size: {cut}")
    print("First 20 partition labels:", partition[:20], "...")

    # 5) Store results as a dictionary in .pkl
    store_partition_results_as_pkl(partition, filename="hypergraph_partition.pkl")

if __name__ == "__main__":
    main()