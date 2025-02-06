import os
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch_geometric.utils import scatter

import time
import wandb
from tqdm import tqdm
from collections import Counter

import sys
sys.path.insert(1, 'data/')
from pyg_dataset import NetlistDataset

sys.path.append("models/layers/")
# Import the modified GNN model with VGAE functionality.
# Set use_vgae=True to activate the unsupervised branch.
from models.model_att_vgae import GNN_node

from sklearn.metrics import accuracy_score, precision_score, recall_score

# Function to compute accuracy, precision, and recall
def compute_metrics(true_labels, predicted_labels):
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='binary')
    recall = recall_score(true_labels, predicted_labels, average='binary')
    return accuracy, precision, recall

### hyperparameters ###
test = False           # if only test but not train
restart = False        # if restart training
reload_dataset = False # if reload already processed dataset

if test:
    restart = True

model_type = "dehnn"  # one of ["dehnn", "dehnn_att", "digcn", "digat"]
num_layer = 3         # note: large number may cause OOM
num_dim = 16          # note: large number may cause OOM
vn = True           # use virtual node or not
trans = False         # use transformer or not
aggr = "add"          # aggregation method: "add" or "max"
device = "cpu"       # device: "cuda" or "cpu"
learning_rate = 0.001

# New hyperparameter: weight for the VGAE unsupervised loss
lambda_vgae = 0.1

# Load dataset and prepare heterogeneous data objects
if not reload_dataset:
    dataset = NetlistDataset(data_dir="data/superblue", load_pe=True, pl=True, processed=True, load_indices=None)
    h_dataset = []
    for data in tqdm(dataset):
        num_instances = data.node_features.shape[0]
        data.num_instances = num_instances
        data.edge_index_sink_to_net[1] = data.edge_index_sink_to_net[1] - num_instances
        data.edge_index_source_to_net[1] = data.edge_index_source_to_net[1] - num_instances
        
        out_degrees = data.net_features[:, 1]
        mask = (out_degrees < 3000)
        mask_edges = mask[data.edge_index_source_to_net[1]] 
        filtered_edge_index_source_to_net = data.edge_index_source_to_net[:, mask_edges]
        data.edge_index_source_to_net = filtered_edge_index_source_to_net

        mask_edges = mask[data.edge_index_sink_to_net[1]] 
        filtered_edge_index_sink_to_net = data.edge_index_sink_to_net[:, mask_edges]
        data.edge_index_sink_to_net = filtered_edge_index_sink_to_net

        h_data = HeteroData()
        h_data['node'].x = data.node_features
        h_data['net'].x = data.net_features
        
        edge_index = torch.cat([data.edge_index_sink_to_net, data.edge_index_source_to_net], dim=1)
        h_data['node', 'to', 'net'].edge_index, h_data['node', 'to', 'net'].edge_weight = gcn_norm(edge_index, add_self_loops=False)
        h_data['node', 'to', 'net'].edge_type = torch.cat([torch.zeros(data.edge_index_sink_to_net.shape[1]), 
                                                            torch.ones(data.edge_index_source_to_net.shape[1])]).bool()
        h_data['net', 'to', 'node'].edge_index, h_data['net', 'to', 'node'].edge_weight = gcn_norm(edge_index.flip(0), add_self_loops=False)
        
        h_data['design_name'] = data['design_name']
        h_data.num_instances = data.node_features.shape[0]
        variant_data_lst = []
        
        node_demand = data.node_demand
        net_demand = data.net_demand
        net_hpwl = data.net_hpwl
        
        batch = data.batch
        num_vn = len(np.unique(batch))
        vn_node = torch.cat([global_mean_pool(h_data['node'].x, batch), 
                             global_max_pool(h_data['node'].x, batch)], dim=1)

        # Standardize targets
        node_demand = (node_demand - torch.mean(node_demand)) / torch.std(node_demand)
        net_hpwl = (net_hpwl - torch.mean(net_hpwl)) / torch.std(net_hpwl)
        net_demand = (net_demand - torch.mean(net_demand)) / torch.std(net_demand)

        variant_data_lst.append((node_demand, net_hpwl, net_demand, batch, num_vn, vn_node)) 
        h_data['variant_data_lst'] = variant_data_lst
        h_dataset.append(h_data)
        
    torch.save(h_dataset, "h_dataset.pt")
else:
    h_dataset = torch.load("h_dataset.pt")
    
# Initialize (or load) the model.
h_data = h_dataset[0]
if restart:
    model = torch.load(f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt")
else:
    # Set use_vgae=True to activate the variational branch.
    model = GNN_node(num_layer, num_dim, 1, 1, node_dim=h_data['node'].x.shape[1], 
                     net_dim=h_data['net'].x.shape[1], gnn_type=model_type, vn=vn, trans=trans, 
                     aggr=aggr, JK="Normal", use_vgae=True).to(device)

criterion_node = nn.MSELoss()
criterion_net = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

load_data_indices = [idx for idx in range(len(h_dataset))]
# For demonstration, we split indices (adjust as needed)
all_train_indices = load_data_indices[:10]
all_valid_indices = load_data_indices[10:]
all_test_indices = load_data_indices[10:]
best_total_val = None

num_epochs = 10

if not test:
    for epoch in range(num_epochs):
        np.random.shuffle(all_train_indices)
        loss_node_all = 0
        loss_net_all = 0
        val_loss_node_all = 0
        val_loss_net_all = 0
        all_train_idx = 0
        
        # Training loop
        for data_idx in tqdm(all_train_indices):
            data = h_dataset[data_idx]
            for inner_data_idx in range(len(data.variant_data_lst)):
                target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
                optimizer.zero_grad()
                data.batch = batch
                data.num_vn = num_vn
                data.vn = vn_node
                # Unpack model outputs; if VGAE is enabled, we get extra losses.
                outputs = model(data, device)
                if model.use_vgae:
                    node_representation, net_representation, rec_loss, kl_loss = outputs
                else:
                    node_representation, net_representation = outputs

                loss_node = criterion_node(torch.squeeze(node_representation), target_node.to(device))
                loss_net = criterion_net(torch.squeeze(net_representation), target_net_demand.to(device))
                loss = loss_node + loss_net

                if model.use_vgae:
                    loss += lambda_vgae * (rec_loss + kl_loss)

                loss.backward()
                optimizer.step()   
                
                loss_node_all += loss_node.item()
                loss_net_all += loss_net.item()
                all_train_idx += 1
        print(f"Epoch {epoch}: Train Node Loss {loss_node_all/all_train_idx:.4f}, Net Loss {loss_net_all/all_train_idx:.4f}")
    
        all_valid_idx = 0
        for data_idx in tqdm(all_valid_indices):
            data = h_dataset[data_idx]
            for inner_data_idx in range(len(data.variant_data_lst)):
                target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
                data.batch = batch
                data.num_vn = num_vn
                data.vn = vn_node
                outputs = model(data, device)
                if model.use_vgae:
                    node_representation, net_representation, rec_loss, kl_loss = outputs
                else:
                    node_representation, net_representation = outputs
                val_loss_node = criterion_node(torch.squeeze(node_representation), target_node.to(device))
                val_loss_net = criterion_net(torch.squeeze(net_representation), target_net_demand.to(device))
                val_loss_node_all +=  val_loss_node.item()
                val_loss_net_all += val_loss_net.item()
                all_valid_idx += 1
        print(f"Epoch {epoch}: Val Node Loss {val_loss_node_all/all_valid_idx:.4f}, Val Net Loss {val_loss_net_all/all_valid_idx:.4f}")
    
        if (best_total_val is None) or ((loss_node_all/all_train_idx) < best_total_val):
            best_total_val = loss_node_all/all_train_idx
            torch.save(model, f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt")
else:
    all_test_idx = 0
    test_loss_node_all = 0
    test_loss_net_all = 0
    for data_idx in tqdm(all_test_indices):
        data = h_dataset[data_idx]
        for inner_data_idx in range(len(data.variant_data_lst)):
            target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
            data.batch = batch
            data.num_vn = num_vn
            data.vn = vn_node
            outputs = model(data, device)
            if model.use_vgae:
                node_representation, net_representation, rec_loss, kl_loss = outputs
            else:
                node_representation, net_representation = outputs
            test_loss_node = criterion_node(torch.squeeze(node_representation), target_node.to(device))
            test_loss_net = criterion_net(torch.squeeze(net_representation), target_net_demand.to(device))
            test_loss_node_all +=  test_loss_node.item()
            test_loss_net_all += test_loss_net.item()
            all_test_idx += 1
    print("Avg test node demand MSE: ", test_loss_node_all/all_test_idx)
    print("Avg test net demand MSE: ", test_loss_net_all/all_test_idx)