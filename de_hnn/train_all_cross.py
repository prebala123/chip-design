import os
import numpy as np
import pandas as pd
import pickle
import time
from datetime import datetime

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
import wandb
from tqdm import tqdm
from collections import Counter

import sys
sys.path.insert(1, 'data/')
from pyg_dataset import NetlistDataset

sys.path.append("models/layers/")
from models.model_att import GNN_node
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Function to compute accuracy, precision, and recall
def compute_metrics(true_labels, predicted_labels):
    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Precision
    precision = precision_score(true_labels, predicted_labels, average='binary')
    
    # Recall
    recall = recall_score(true_labels, predicted_labels, average='binary')
    
    return accuracy, precision, recall

### hyperparameter ###
test = False # if only test but not train
restart = False # if restart training
reload_dataset = True # if reload already processed h_dataset

if test:
    restart = True

prediction = 'congestion' # one of ['congestion', 'demand']
model_type = "dehnn" #this can be one of ["dehnn", "dehnn_att", "digcn", "digat"] "dehnn_att" might need large memory usage
num_layer = 2 #large number will cause OOM
num_dim = 16 #large number will cause OOM
vn = True #use virtual node or not
trans = False #use transformer or not
aggr = "add" #use aggregation as one of ["add", "max"]
device = "cpu" #use cuda or cpu
learning_rate = 0.001
num_epochs = 500

if not reload_dataset:
    dataset = NetlistDataset(data_dir="data/superblue", load_pe = True, pl = True, processed = True, load_indices=None)
    h_dataset = []
    for data in tqdm(dataset):
        num_instances = data.node_features.shape[0]
        data.num_instances = num_instances
        data.edge_index_sink_to_net[1] = data.edge_index_sink_to_net[1] - num_instances
        data.edge_index_source_to_net[1] = data.edge_index_source_to_net[1] - num_instances
        
        # print(data.net_features.shape)
        out_degrees = data.net_features[:, 0]
        mask = (out_degrees < 3000)
        mask_edges = mask[data.edge_index_source_to_net[1]] 
        filtered_edge_index_source_to_net = data.edge_index_source_to_net[:, mask_edges]
        data.edge_index_source_to_net = filtered_edge_index_source_to_net

        mask_edges = mask[data.edge_index_sink_to_net[1]] 
        filtered_edge_index_sink_to_net = data.edge_index_sink_to_net[:, mask_edges]
        data.edge_index_sink_to_net = filtered_edge_index_sink_to_net

        h_data = HeteroData()
        h_data['node'].x = data.node_features
        # print(data.node_features.shape)
        h_data['net'].x = data.net_features.float()
        print(h_data['node'].x.shape)
        # print(h_data['node'].x.type())
        print(h_data['net'].x.shape)
        # print(h_data['net'].x.type())

        
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
        
        batch = data.batch
        num_vn = len(np.unique(batch))
        vn_node = torch.concat([global_mean_pool(h_data['node'].x, batch), 
                global_max_pool(h_data['node'].x, batch)], dim=1)

        # node_demand = (node_demand - torch.mean(node_demand)) / torch.std(node_demand)
        # net_hpwl = (net_hpwl - torch.mean(net_hpwl)) / torch.std(net_hpwl)
        # net_demand = (net_demand - torch.mean(net_demand))/ torch.std(net_demand)

        variant_data_lst.append((node_demand, net_hpwl, net_demand, batch, num_vn, vn_node)) 
        h_data['variant_data_lst'] = variant_data_lst
        h_dataset.append(h_data)
        
    torch.save(h_dataset, "h_dataset.pt")
    
else:
    dataset = torch.load("h_dataset.pt")
    h_dataset = []
    for data in dataset:
        h_dataset.append(data)
    
sys.path.append("models/layers/")

h_data = h_dataset[0]
if restart:
    model = torch.load(f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt")
else:
    model = GNN_node(num_layer, num_dim, 2, 2, node_dim = h_data['node'].x.shape[1], net_dim = h_data['net'].x.shape[1], gnn_type=model_type, vn=vn, trans=trans, aggr=aggr, JK="Normal").to(device)

criterion_node = nn.CrossEntropyLoss()
criterion_net = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate,  weight_decay=0.01)
load_data_indices = [idx for idx in range(len(h_dataset))]
all_train_indices, all_valid_indices, all_test_indices = load_data_indices[:7] + load_data_indices[8:11], load_data_indices[7:8], load_data_indices[7:8]
best_total_val = None

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
filepath = f"baselines/{model_type}_{num_epochs}_{num_layer}_{num_dim}_{vn}_{trans}_{timestamp}_classify.csv"
with open(filepath, 'a') as f:
    f.write('Epoch,tp,fp,tn,fn,p,r,fsc,m,Time\n')

if not test:
    start_time = time.time()
    for epoch in range(num_epochs):
        np.random.shuffle(all_train_indices)
        loss_node_all = 0
        loss_net_all = 0
        val_loss_node_all = 0
        val_loss_net_all = 0
        
        all_train_idx = 0
        for data_idx in tqdm(all_train_indices):
            data = h_dataset[data_idx]
            for inner_data_idx in range(len(data.variant_data_lst)):
                target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
                optimizer.zero_grad()
                data.batch = batch
                data.num_vn = num_vn
                data.vn = vn_node
                node_representation, net_representation = model(data, device)
                node_representation = torch.squeeze(node_representation)
                net_representation = torch.squeeze(net_representation)
    
                loss_node = criterion_node(node_representation, target_node.to(device))
                loss_net = criterion_net(net_representation, target_net_demand.to(device))
                loss = loss_node# + loss_net
                loss.backward()
                optimizer.step()   
    
                loss_node_all += loss_node.item()
                loss_net_all += loss_net.item()
                all_train_idx += 1
        print(loss_node_all/all_train_idx, loss_net_all/all_train_idx)
    
        all_valid_idx = 0
        for data_idx in tqdm(all_valid_indices):
            data = h_dataset[data_idx]
            for inner_data_idx in range(len(data.variant_data_lst)):
                target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
                data.batch = batch
                data.num_vn = num_vn
                data.vn = vn_node
                node_representation, net_representation = model(data, device)
                node_representation = torch.squeeze(node_representation)
                net_representation = torch.squeeze(net_representation)
                
                val_loss_node = criterion_node(node_representation, target_node.to(device))
                val_loss_net = criterion_net(net_representation, target_net_demand.to(device))
                val_loss_node_all +=  val_loss_node.item()
                val_loss_net_all += val_loss_net.item()
                all_valid_idx += 1

                outs = node_representation.detach().numpy()
                pred_vals = np.array([0 if i > j else 1 for i, j in outs])
                real_vals = target_node.numpy()
                tp, fp, tn, fn = 0, 0, 0, 0
                for i, j in zip(pred_vals, real_vals):
                    if i == 1 and j == 1:
                        tp += 1
                    elif i == 1 and j == 0:
                        fp += 1
                    elif i == 0 and j == 1:
                        fn += 1
                    else:
                        tn += 1
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                fsc = (2 * p * r) / (p + r)
                m = np.mean(pred_vals)
                print(f' {m}')
                print(' ', tp, fp, '\n', fn, tn)
                print(f' Precision: {p}, Recall: {r}')

        print(val_loss_node_all/all_valid_idx, val_loss_net_all/all_valid_idx)
        print(f'Epoch {epoch+1}, {round(time.time() - start_time, 2)}s\n')
    
        if (best_total_val is None) or ((loss_node_all/all_train_idx) < best_total_val):
            best_total_val = loss_node_all/all_train_idx
            torch.save(model, f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_classify.pt")

        with open(filepath, 'a') as f:
            f.write(f'{epoch+1},{tp},{fp},{tn},{fn},{p},{r},{fsc},{m},{round(time.time()-start_time, 2)}\n')
            # f.write(f'{epoch+1},{loss_node_all/all_train_idx},{loss_net_all/all_train_idx},{val_loss_node_all/all_valid_idx},{val_loss_net_all/all_valid_idx},{round(time.time()-start_time, 2)}\n')
        
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
            node_representation, net_representation = model(data, device)
            node_representation = torch.squeeze(node_representation)
            net_representation = torch.squeeze(net_representation)
            
            test_loss_node = criterion_node(node_representation, target_node.to(device))
            test_loss_net = criterion_net(net_representation, target_net_demand.to(device))
            test_loss_node_all +=  test_loss_node.item()
            test_loss_net_all += test_loss_net.item()
            all_test_idx += 1
    print("avg test node demand mse: ", test_loss_node_all/all_test_idx)
    print("avg test net demand mse: ", test_loss_net_all/all_test_idx)