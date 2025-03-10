import os
import numpy as np
import pickle
import csv
# import logging

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import Dataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from torch_geometric.utils import scatter

import time
# import wandb
from tqdm import tqdm
from collections import Counter

import sys
# sys.path.insert(1, 'data/')
from pyg_dataset_sub import NetlistDataset as NetlistDatasetSub # type: ignore
# from pyg_dataset import NetlistDataset # type: ignore

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from models.model_att import GNN_node

import matplotlib.pyplot as plt

# Function to generate outputs without backpropagating the errors
def evaluate_model(indices, dataset, model, vn=False):
    criterion_node = nn.MSELoss()
    criterion_net = nn.MSELoss()
    idx = 0
    node_loss_all = 0
    net_loss_all = 0
    with torch.no_grad():
        for data_idx in indices:
            data = dataset[data_idx].to(device)
            for inner_data_idx in range(len(data.variant_data_lst)):
                if vn:
                    target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
                    data.batch = batch
                    data.num_vn = num_vn
                    data.vn = vn_node
                else:
                    target_node, target_net_hpwl, target_net_demand = data.variant_data_lst[inner_data_idx]

                node_representation, net_representation = model(data, device)
                node_representation = torch.squeeze(node_representation)
                net_representation = torch.squeeze(net_representation)
                
                node_loss = criterion_node(node_representation, target_node.to(device))
                net_loss = criterion_net(net_representation, target_net_demand.to(device))
                node_loss_all +=  node_loss.item()
                net_loss_all += net_loss.item()
                idx += 1

    return node_loss_all/idx, net_loss_all/idx

def save_losses_to_csv(train_losses, val_losses, test_losses, filename):
    # Write to CSV file
    os.makedirs('loss_csvs', exist_ok=True)

    path = os.path.join('loss_csvs', filename + '.csv')
    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # Write headers
        headers = ['Train Loss', 'Validation Loss', 'Test Loss']
        writer.writerow(headers)
        
        # Write rows
        for row in zip(train_losses, val_losses, test_losses):
            writer.writerow(row)

# Function to plot loss curves
def plot_loss_curves(train_losses, val_losses, test_losses, filename):
    """
    Plots training and validation loss curves over epochs and saves the plot.

    Parameters:
    train_losses (list of float): Training loss values.
    val_losses (list of float): Validation loss values.
    save_path (str): Path to save the plot.
    """
    if len(train_losses) != len(val_losses):
        raise ValueError("Both lists must have the same length.")
    
    # Ensure the directory exists
    os.makedirs('loss_plots', exist_ok=True)
    
    epochs = list(range(1, len(train_losses) + 1))

    train_losses = [loss**(1/2) for loss in train_losses]
    val_losses = [loss**(1/2) for loss in val_losses]
    test_losses = [loss**(1/2) for loss in test_losses]

    if filename.split('_')[1] == 'node':
        plotname = 'Node'
    else:
        plotname = 'Net'

    save_losses_to_csv(train_losses, val_losses, test_losses, filename)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.plot(epochs, test_losses, label='Test Loss')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(plotname + ' RMSE Loss Curves')
    plt.legend()
    plt.grid(True)

    # Save plot
    save_path = os.path.join('loss_plots', filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved at {save_path}")

# Function to condense various subgraph pkl files to single object
def create_h_dataset():
    dataset = NetlistDatasetSub(data_dir=os.path.join('..', '..', 'data', 'superblue'), load_pe = True, pl = True, processed = True, load_indices=None)
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
        
        edge_index = torch.concat([data.edge_index_sink_to_net, data.edge_index_source_to_net], dim=1)
        h_data['node', 'to', 'net'].edge_index, h_data['node', 'to', 'net'].edge_weight = gcn_norm(edge_index, add_self_loops=False)
        h_data['node', 'to', 'net'].edge_type = torch.concat([torch.zeros(data.edge_index_sink_to_net.shape[1]), torch.ones(data.edge_index_source_to_net.shape[1])]).bool()
        h_data['net', 'to', 'node'].edge_index, h_data['net', 'to', 'node'].edge_weight = gcn_norm(edge_index.flip(0), add_self_loops=False)
        
        h_data['design_name'] = data['design_name']
        h_data.num_instances = data.node_features.shape[0]
        variant_data_lst = []
        
        node_demand = data.node_demand
        net_demand = data.net_demand
        net_hpwl = data.net_hpwl

        # node_demand = (node_demand - torch.mean(node_demand)) / torch.std(node_demand)
        # net_hpwl = (net_hpwl - torch.mean(net_hpwl)) / torch.std(net_hpwl)
        # net_demand = (net_demand - torch.mean(net_demand))/ torch.std(net_demand)

        variant_data_lst.append((node_demand, net_hpwl, net_demand))

        h_data['variant_data_lst'] = variant_data_lst
        h_dataset.append(h_data)
        
    torch.save(h_dataset, "h_dataset_sub.pt")
    print('h_dataset_sub.pt saved')

    return h_dataset
    
def train(model, learning_rate, epochs, h_dataset, checkpoint, sample_prop=0.6):
    criterion_node = nn.MSELoss()
    criterion_net = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    sub_train_indices = [idx for idx in range(len(h_dataset)) if '18' not in h_dataset[idx]['design_name'] and '19' not in h_dataset[idx]['design_name']]
    sub_valid_indices = [idx for idx in range(len(h_dataset)) if '18' in h_dataset[idx]['design_name']]
    sub_test_indices =  [idx for idx in range(len(h_dataset)) if '19' in h_dataset[idx]['design_name']]
    best_total_val = None

    sub_train_losses = {'node': [], 'net': []}
    sub_valid_losses = {'node': [], 'net': []}
    sub_test_losses = {'node': [], 'net': []}
    best_losses = {'node': [], 'net': []}
    
    for epoch in tqdm(range(epochs)):

        # np.random.shuffle(sub_train_indices)
        sample_size = int(len(sub_train_indices) * sample_prop)
        sub_train_indices_sample = np.random.choice(sub_train_indices, size=sample_size, replace=False)

        model.train()
        sub_train_idx = 0
        sub_node_train_loss = 0
        sub_net_train_loss = 0
        for data_idx in sub_train_indices_sample:
            data = h_dataset[data_idx].to(device)
            for inner_data_idx in range(len(data.variant_data_lst)):
                target_node, _, target_net_demand = data.variant_data_lst[inner_data_idx]

                optimizer.zero_grad()
                node_representation, net_representation = model(data, device)
                node_representation = torch.squeeze(node_representation)
                net_representation = torch.squeeze(net_representation)

                loss_node = criterion_node(node_representation, target_node.to(device))
                loss_net = criterion_net(net_representation, target_net_demand.to(device))
                loss = loss_node + loss_net 
                loss.backward()
                optimizer.step()   

                sub_node_train_loss += loss_node.item()
                sub_net_train_loss += loss_net.item()
                sub_train_idx += 1

            # scheduler.step()

        sub_train_losses['node'].append(sub_node_train_loss/sub_train_idx)
        sub_train_losses['net'].append(sub_net_train_loss/sub_train_idx)

        model.eval()
        node_loss, net_loss = evaluate_model(sub_valid_indices, h_dataset, model)
        sub_valid_losses['node'].append(node_loss)
        sub_valid_losses['net'].append(net_loss)

        node_loss, net_loss = evaluate_model(sub_test_indices, h_dataset, model)
        sub_test_losses['node'].append(node_loss)
        sub_test_losses['net'].append(net_loss)

        model_family = f"{model_type}_{num_layer}_{num_dim}_{epochs}_{int(learning_rate*1000)}_{int(sample_prop*100)}"
        os.makedirs(os.path.join('fitted_models', f"{model_family}_model"), exist_ok=True)

        if (best_total_val is None) or ((sub_valid_losses['node'][-1] + sub_valid_losses['net'][-1]) < best_total_val):
            best_total_val = sub_valid_losses['node'][-1] + sub_valid_losses['net'][-1]
            best_losses['node'] = [sub_train_losses['node'][-1]**(1/2), sub_valid_losses['node'][-1]**(1/2), sub_test_losses['node'][-1]**(1/2)]
            best_losses['net'] = [sub_train_losses['net'][-1]**(1/2), sub_valid_losses['net'][-1]**(1/2), sub_test_losses['net'][-1]**(1/2)]
            torch.save(model, os.path.join('fitted_models', f"{model_family}_model", f"{model_family}_model.pt"))

        if checkpoint and (epoch+1)%10==0:
            filepath = os.path.join(f"{model_family}_model", f"{model_family}_{epoch+1}_model.pt")
            torch.save(model, os.path.join('fitted_models', filepath))

    return sub_train_losses, sub_valid_losses, sub_test_losses, best_losses

if __name__ == '__main__':
    ### hyperparameter ###
    test = False # if only test but not train
    restart = False # if restart training
    reload_dataset = True # if reload already processed h_dataset
    checkpoint = False

    model_type = "dehnn" #this can be one of ["dehnn", "dehnn_att", "digcn", "digat"] "dehnn_att" might need large memory usage
    num_layer = 2 #large number will cause OOM
    num_dim = 16 #large number will cause OOM
    vn = False #use virtual node or not
    trans = False #use transformer or not
    aggr = "add" #use aggregation as one of ["add", "max"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 0.00001
    epochs = 1

    sample_prop = 0.8

    if reload_dataset:
        dataset = torch.load('h_dataset_sub.pt')
        h_dataset = []
        for data in dataset:
            h_dataset.append(data)
    else:
        h_dataset = create_h_dataset()

    h_data = h_dataset[0]

    print(f'\nModel Parameters:')
    print(f'    Num Layers: {num_layer}')
    print(f'    Dimension: {num_dim}')
    print(f'    Epochs: {epochs}')
    print(f'    Learning Rate: {learning_rate}')
    print(f'    Sample Proportion: {sample_prop}')

    model = GNN_node(num_layer, num_dim, 1, 1, node_dim = h_data['node'].x.shape[1], net_dim = h_data['net'].x.shape[1], gnn_type=model_type, vn=vn, trans=trans, aggr=aggr, JK="Normal").to(device)

    sub_train_losses, sub_valid_losses, sub_test_losses, best_losses = train(model, learning_rate, epochs, h_dataset, checkpoint, sample_prop)

    print(f"node train loss: {best_losses['node'][0]}")
    print(f"node valid loss: {best_losses['node'][1]}")
    print(f"node test loss: {best_losses['node'][2]}")
    plot_loss_curves(sub_train_losses['node'], sub_valid_losses['node'], sub_test_losses['node'], 
                     filename = f'sub_node_{num_layer}_{num_dim}_{epochs}_{int(learning_rate*1000)}_{int(sample_prop*100)}_losses')
    
    print(f"net train loss: {best_losses['net'][0]}")
    print(f"net valid loss: {best_losses['net'][1]}")
    print(f"net test loss: {best_losses['net'][2]}")
    plot_loss_curves(sub_train_losses['net'], sub_valid_losses['net'], sub_test_losses['net'], 
                     filename = f'sub_net_{num_layer}_{num_dim}_{epochs}_{int(learning_rate*1000)}_{int(sample_prop*100)}_losses')
