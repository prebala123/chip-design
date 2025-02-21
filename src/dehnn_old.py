import numpy as np
import torch
import torch.nn as nn
import pickle
import time
import datetime as datetime

class DEHNNLayer(nn.Module):
    def __init__(self, node_in_features, edge_in_features, vn_features, hidden_features):
        super(DEHNNLayer, self).__init__()
        self.node_mlp1 = nn.Sequential(nn.Linear(edge_in_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, edge_in_features))
        
        self.edge_mlp2 = nn.Sequential(nn.Linear(node_in_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, node_in_features))
        
        self.edge_mlp3 = nn.Sequential(nn.Linear(2 * node_in_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, 2 * node_in_features))

        self.node_to_virtual_mlp = nn.Sequential(nn.Linear(node_in_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, vn_features))
        
        self.virtual_to_higher_virtual_mlp = nn.Sequential(nn.Linear(vn_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, vn_features))
        
        self.higher_virtual_to_virtual_mlp = nn.Sequential(nn.Linear(vn_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, vn_features))
        
        self.virtual_to_node_mlp = nn.Sequential(nn.Linear(vn_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, edge_in_features))


    def forward(self, node_features, edge_features, vn_features, super_vn_features, hypergraph):

        # Node Update
        transformed_edge_features = self.node_mlp1(edge_features)
        updated_node_features = torch.matmul(hypergraph.incidence_matrix, transformed_edge_features)

        # Edge Update
        transformed_node_features = self.edge_mlp2(node_features)
        driver_features = torch.matmul(hypergraph.driver_matrix, transformed_node_features)
        sink_features = torch.matmul(hypergraph.sink_matrix, transformed_node_features)
        updated_edge_features = torch.cat([driver_features, sink_features], dim=1)
        updated_edge_features = self.edge_mlp3(updated_edge_features)
        
        # First Level VN Update
        node_to_virtual_features = self.node_to_virtual_mlp(node_features)
        updated_vn_features = torch.matmul(hypergraph.vn_matrix, node_to_virtual_features)
        updated_vn_features += self.higher_virtual_to_virtual_mlp(super_vn_features)

        # Top Level VN Update
        virtual_to_higher_virtual_features = self.virtual_to_higher_virtual_mlp(vn_features)
        updated_super_vn_features = torch.sum(virtual_to_higher_virtual_features, dim=0)

        # VN to node update
        virtual_to_node_features = self.virtual_to_node_mlp(vn_features)
        propagated_features = torch.matmul(hypergraph.vn_matrix.T, virtual_to_node_features)
        updated_node_features += propagated_features

        return updated_node_features, updated_edge_features, updated_vn_features, updated_super_vn_features


class DEHNN(nn.Module):
    def __init__(self, num_layers, node_in_features, edge_in_features, hidden_features=16):
        super(DEHNN, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Create multiple layers for DEHNN
        vn_in_features = node_in_features
        for i in range(num_layers):
            self.layers.append(DEHNNLayer(node_in_features, edge_in_features, vn_in_features, hidden_features))
            node_in_features, edge_in_features = edge_in_features, node_in_features
            edge_in_features *= 2
        
        self.output_layer = nn.Sequential(nn.Linear(node_in_features, hidden_features),
                                       nn.ReLU(),
                                       nn.Linear(hidden_features, 1))

    def forward(self, node_features, edge_features, vn_features, super_vn_features, hypergraph):
        # Pass through each layer
        for layer in self.layers:
            node_features, edge_features, vn_features, super_vn_features = layer(node_features, edge_features, vn_features, super_vn_features, hypergraph)
        
        # Output prediction for nodes
        output = self.output_layer(node_features)
        return output

class Hypergraph:
    def __init__(self, incidence_matrix, driver_matrix, sink_matrix, vn_matrix):
        self.incidence_matrix = incidence_matrix
        self.driver_matrix = driver_matrix
        self.sink_matrix = sink_matrix
        self.vn_matrix = vn_matrix

def open_chip(i, device):
    path = f'../data/superblue/superblue_{i}/'

    with open(path + f'bipartite.pkl', 'rb') as f:
        bipartite = pickle.load(f)

    with open(path + f'degree.pkl', 'rb') as f:
        degree = pickle.load(f)

    with open(path + f'eigen.10.pkl', 'rb') as f:
        eigen = pickle.load(f)

    with open(path + f'metis_part_dict.pkl', 'rb') as f:
        metis = pickle.load(f)

    with open(path + f'net_hpwl.pkl', 'rb') as f:
        hpwl = pickle.load(f)

    with open(path + f'node_features.pkl', 'rb') as f:
        node_feats = pickle.load(f)

    with open(path + f'node_neighbor_features.pkl', 'rb') as f:
        node_neighbor_feats = pickle.load(f)

    with open(path + f'targets.pkl', 'rb') as f:
        targets = pickle.load(f)

    incidence_matrix = torch.sparse_coo_tensor(torch.tensor(np.array([bipartite['instance_idx'], bipartite['net_idx']])), torch.ones(bipartite['edge_dir'].shape), dtype=torch.float).to(device)

    driver_idx = bipartite['edge_dir'] == 1
    driver_row = bipartite['instance_idx'][driver_idx]
    driver_col = bipartite['net_idx'][driver_idx]
    driver_dir = bipartite['edge_dir'][driver_idx]
    driver_matrix = torch.sparse_coo_tensor(torch.tensor(np.array([driver_col, driver_row])), torch.ones(driver_dir.shape), dtype=torch.float).to(device)

    sink_idx = bipartite['edge_dir'] == 0
    sink_row = bipartite['instance_idx'][sink_idx]
    sink_col = bipartite['net_idx'][sink_idx]
    sink_dir = bipartite['edge_dir'][sink_idx]
    sink_matrix = torch.sparse_coo_tensor(torch.tensor(np.array([sink_col, sink_row])), torch.ones(sink_dir.shape), dtype=torch.float).to(device)

    num_nodes = node_feats['num_instances']
    node_features = node_feats['instance_features'][:, 2:]
    node_features = torch.tensor(node_features, dtype=torch.float)
    cell_degrees = torch.tensor(degree['cell_degrees'])
    net_degrees = torch.tensor(degree['net_degrees'])
    node_features = torch.cat([node_features, cell_degrees.unsqueeze(dim = 1)], dim = 1)
    edge_features = net_degrees.unsqueeze(dim = 1)

    evects_node = torch.tensor(eigen['evects'][:num_nodes])
    evects_net = torch.tensor(eigen['evects'][num_nodes:])
    node_features = torch.cat([node_features, evects_node], dim=1)
    edge_features = torch.cat([edge_features, evects_net], dim=1)

    # pd = torch.Tensor(node_neighbor_feats['pd'])
    neighbor_list = torch.Tensor(node_neighbor_feats['neighbor'])
    # node_features = torch.cat([node_features, pd, neighbor_list], dim = 1)
    node_features = torch.cat([node_features, neighbor_list], dim = 1)

    node_features = node_features.to(device)
    edge_features = edge_features.to(device)

    num_nodes, num_node_features = node_features.shape
    num_edges, num_edge_features = edge_features.shape
    demand = targets['demand'].reshape(num_nodes,1)
    demand = torch.tensor(demand, dtype=torch.float).to(device)
    demand = (demand - torch.mean(demand)) / torch.std(demand)
    # wire = hpwl['hpwl'].reshape(num_edges,1)
    # wire = torch.tensor(wire, dtype=torch.float).to(device)

    vn_row = []
    vn_col = []

    for k, v in metis.items():
        if k < num_nodes:
            vn_row.append(v)
            vn_col.append(k)

    vn_matrix = torch.sparse_coo_tensor(torch.tensor(np.array([vn_row, vn_col])), torch.ones(len(vn_row)), dtype=torch.float).to(device)

    num_vn = vn_matrix.shape[0]
    num_vn_features = num_node_features
    vn_features = torch.zeros((num_vn, num_vn_features), dtype=torch.float).to(device)
    super_vn_features = torch.zeros(num_vn_features, dtype=torch.float).to(device)

    hypergraph = Hypergraph(incidence_matrix, driver_matrix, sink_matrix, vn_matrix)

    return node_features, edge_features, vn_features, super_vn_features, hypergraph, demand

num_node_features = 27
num_edge_features = 11
num_layers = 2
hidden_dim = 16
epochs = 1000
device = 'cpu'
s = 1

train_idx = [1, 2, 3, 5, 6, 7, 9, 11, 14, 16]
valid_idx = 19

all_designs = [open_chip(i, device) for i in train_idx]

valid_node_features, valid_edge_features, valid_vn_features, valid_super_vn_features, valid_hypergraph, valid_demand = open_chip(valid_idx, device)

# Initialize DE-HNN model
np.random.seed(s)
model = DEHNN(num_layers=num_layers, node_in_features=num_node_features, edge_in_features=num_edge_features, hidden_features=hidden_dim).to(device)

# Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

filepath = f'baseline_{s}.csv'
with open(filepath, 'w') as f:
    f.write('Epoch,Train_Loss,Valid_Loss,Time\n')

best_valid = float('inf')

start_time = time.time()
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0
    # for i in train_idx:
    for i in range(len(train_idx)):
        node_features, edge_features, vn_features, super_vn_features, hypergraph, demand = all_designs[i]
        # node_features, edge_features, vn_features, super_vn_features, hypergraph, demand = open_chip(i, device)

        output = model(node_features, edge_features, vn_features, super_vn_features, hypergraph)

        loss = criterion(output, demand)
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()

    # Print loss
    model.eval()
    output = model(valid_node_features, valid_edge_features, valid_vn_features, valid_super_vn_features, valid_hypergraph)
    valid_loss = criterion(output, valid_demand)
    # if epoch % 10 == 9:
    #     print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {valid_loss.item():.4f}')
    print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {total_loss/10:.4f}, Validation Loss: {valid_loss.item():.4f}, Time: {round(time.time()-start_time, 2)}s')

    with open(filepath, 'a') as f:
        f.write(f'{epoch+1},{total_loss/10},{valid_loss.item()},{round(time.time()-start_time, 2)}\n')

    if valid_loss.item() < best_valid:
        best_valid = valid_loss.item()
        torch.save(model, 'demand_superblue.pt')