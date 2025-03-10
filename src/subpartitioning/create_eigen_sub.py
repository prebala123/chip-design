import os
import pickle
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix, diags, identity
from cupyx.scipy.sparse.linalg import eigsh
# from scipy.sparse import csr_matrix, diags, identity
# from scipy.sparse.linalg import eigsh
from tqdm import tqdm

# Build the full graph's adjacency list from the bipartite data.
def build_adjacency(bipartite_dict, node_features_dict):
    """
    Build the full graph’s adjacency list.
    Instance nodes: indices 0..(M-1)
    Net nodes: indices M..(M+N-1) (net indices are offset by M)
    """
    instance_idx = bipartite_dict["instance_idx"]
    net_idx = bipartite_dict["net_idx"]
    edge_index = bipartite_dict["edge_index"]

    num_instances = node_features_dict['num_instances']
    num_nets = node_features_dict['num_nets']
    total_nodes = num_instances + num_nets

    adjacency = [[] for _ in range(total_nodes)]
    for col in range(edge_index.shape[1]):
        inst = instance_idx[col]
        net = net_idx[col] + num_instances  # offset net indices
        adjacency[inst].append(net)
        adjacency[net].append(inst)
    return adjacency

# Extract an induced subgraph and return a mapping from local to global indices.
def extract_subgraph_with_mapping(adjacency, node_indices):
    """
    Extract the induced subgraph from the full graph for the given global node_indices.
    Returns a tuple (sub_adj, mapping) where:
      - sub_adj is the adjacency list for the subgraph (local indices 0 to |S|-1),
      - mapping is a dict mapping local index -> original global index.
    """
    sorted_nodes = sorted(set(node_indices))
    mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(sorted_nodes)}
    reverse_mapping = {old_idx: new_idx for new_idx, old_idx in mapping.items()}
    sub_adj = [[] for _ in range(len(sorted_nodes))]
    for old_idx in sorted_nodes:
        new_idx = reverse_mapping[old_idx]
        sub_adj[new_idx] = [reverse_mapping[nb] for nb in adjacency[old_idx] if nb in reverse_mapping]
    return sub_adj, mapping

# Compute the top k eigenpairs (eigenvalues and eigenvectors) of the normalized Laplacian using CuPy.
def compute_top_k_eigenvectors(adj, k=10):
    """
    Given an adjacency list 'adj' for a subgraph, compute the normalized Laplacian:
        L = I - D^{-1/2} A D^{-1/2}
    using a sparse CSR matrix, then compute the top k eigenpairs (largest algebraic eigenvalues)
    using cupyx.scipy.sparse.linalg.eigsh.
    
    Returns:
      eigenvalues (numpy.ndarray) of shape (k,)
      eigenvectors (numpy.ndarray) of shape (n, k)
    """
    n = len(adj)
    rows = []
    cols = []
    data = []
    for i, neighbors in enumerate(adj):
        for j in neighbors:
            rows.append(i)
            cols.append(j)
            data.append(1)
    # Build a sparse CSR matrix on the GPU using cupy.
    rows_cp = cp.array(rows) # cp not np
    cols_cp = cp.array(cols) # cp not np
    data_cp = cp.array(data, dtype=np.float64) # cp not np
    A = csr_matrix((data_cp, (rows_cp, cols_cp)), shape=(n, n))
    
    # Compute degree vector.
    deg = cp.array(A.sum(axis=1)).flatten() # cp not np
    inv_sqrt_deg = cp.where(deg > 0, 1 / cp.sqrt(deg), 0)
    D_inv_sqrt = diags(inv_sqrt_deg)
    I = identity(n, format='csr')
    # Normalized Laplacian.
    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    
    if n < k:
        k = n
    # "LA" selects the largest algebraic eigenvalues.
    eigenvalues, eigenvectors = eigsh(L, k=k, which='LA')
 
    # Convert results to numpy arrays.
    eigenvalues = cp.asnumpy(eigenvalues)
    eigenvectors = cp.asnumpy(eigenvectors)
    return eigenvalues, eigenvectors

# Process each design folder.
def process_design_folder(design_folder):
    """
    For a given design folder (e.g., superblue_18), load the required files and for each subpartition
    in subpartitions100.pkl, compute:
      - Top 10 eigenpairs of the subgraph (using cupyx eigsh)
      - The degree vector for the subgraph, then split it into cell_degrees and net_degrees
        using num_instances.
    Save the eigenpairs into a folder 'subpartitions_eigen' with files named 'label_eigen.10.pkl'
    and the degree vectors into a folder 'subpartitions_degree' with files named 'label_degree.pkl'.
    """
    # Paths to required files.
    bipartite_fp = os.path.join(design_folder, "bipartite.pkl")
    node_features_fp = os.path.join(design_folder, "node_features.pkl")
    subpartitions_fp = os.path.join(design_folder, "subpartitions100.pkl")
    
    # Check file existence.
    if not (os.path.exists(bipartite_fp) and os.path.exists(node_features_fp) and os.path.exists(subpartitions_fp)):
        print(f"Skipping {design_folder}: missing required files.")
        return
    
    # Load files.
    with open(bipartite_fp, 'rb') as f:
        bipartite_dict = pickle.load(f)
    with open(node_features_fp, 'rb') as f:
        node_features_dict = pickle.load(f)
    with open(subpartitions_fp, 'rb') as f:
        subpartitions_dict = pickle.load(f)
    
    num_instances = node_features_dict['num_instances']
    
    # Build the full graph's adjacency list.
    adjacency = build_adjacency(bipartite_dict, node_features_dict)
    
    # Prepare output directories.
    eigen_dir = os.path.join(design_folder, "subpartitions_eigen")
    degree_dir = os.path.join(design_folder, "subpartitions_degree")
    os.makedirs(eigen_dir, exist_ok=True)
    os.makedirs(degree_dir, exist_ok=True)
    
    # Process each subpartition.
    for label, node_list in tqdm(subpartitions_dict.items()):
        eigen_output_fp = os.path.join(eigen_dir, f"{label}_eigen.10.pkl")
        degree_output_fp = os.path.join(degree_dir, f"{label}_degree.pkl")

        if os.path.exists(eigen_output_fp) and os.path.exists(degree_output_fp):
            print(f'Subpartition {label} already processed')
            continue

        # Extract subgraph and mapping.
        sub_adj, mapping = extract_subgraph_with_mapping(adjacency, node_list)
        
        # Compute top 10 eigenpairs.
        eigenvals, eigenvecs = compute_top_k_eigenvectors(sub_adj, k=10)
        eigen_data = {"evals": eigenvals, "evects": eigenvecs}
        with open(eigen_output_fp, 'wb') as f:
            pickle.dump(eigen_data, f)
        
        # Compute the degree vector for the subgraph.
        # Local degree: simply the length of each neighbor list in sub_adj.
        degree_vector = np.array([len(neighbors) for neighbors in sub_adj])
        
        # Use the mapping to determine the global index for each local node.
        cell_degrees = []
        net_degrees = []
        for local_idx, deg in enumerate(degree_vector):
            global_idx = mapping[local_idx]
            if global_idx < num_instances:
                cell_degrees.append(deg)
            else:
                net_degrees.append(deg)
        degree_data = {"cell_degrees": cell_degrees, "net_degrees": net_degrees}
        with open(degree_output_fp, 'wb') as f:
            pickle.dump(degree_data, f)
        
        print(f"Processed subpartition {label}: saved eigen data to {eigen_output_fp} and degree data to {degree_output_fp}")

def process_all_designs(superblue_folder):
    """
    Loop through each design folder (e.g., superblue_x) in the superblue_folder,
    and process each using process_design_folder.
    """
    for folder in os.listdir(superblue_folder):
        design_folder = os.path.join(superblue_folder, folder)
        if os.path.isdir(design_folder):
            print(f"Processing design folder: {design_folder}")
            process_design_folder(design_folder)

def compute_single_model_eigenvalues(design_num):
    design_folder = f'superblue_{design_num}'
    
    bipartite_fp = os.path.join('superblue', design_folder, "bipartite.pkl")
    node_features_fp = os.path.join('superblue', design_folder, "node_features.pkl")
    subpartitions_fp = os.path.join(design_folder, "subpartitions100.pkl")
    
    # Load files.
    with open(bipartite_fp, 'rb') as f:
        bipartite_dict = pickle.load(f)
    with open(node_features_fp, 'rb') as f:
        node_features_dict = pickle.load(f)
    
    # Build the full graph's adjacency list.
    adjacency = build_adjacency(bipartite_dict, node_features_dict)

    eigen_output_fp = os.path.join('superblue', design_folder, f"eigen.10_recomputed.pkl")

    eigenvals, eigenvecs = compute_top_k_eigenvectors(adjacency, k=10)
    eigen_data = {"evals": eigenvals, "evects": eigenvecs}
    with open(eigen_output_fp, 'wb') as f:
        pickle.dump(eigen_data, f)

    print(f'Recomputed eigenvalues and eigenvectors for Superblue {design_num}')


if __name__ == "__main__":
    # Set the main superblue folder path.
    superblue_folder = os.path.join('..', '..', 'data', 'superblue')
    process_all_designs(superblue_folder)
    # compute_single_model_eigenvalues('18')

    
