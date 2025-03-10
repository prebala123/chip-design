# Import packages
import numpy as np
import time
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import HeteroData
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import global_mean_pool, global_max_pool

from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

from tqdm import tqdm
import sys
from pyg_dataset import NetlistDataset

sys.path.append("models/layers/")
from models.model_att import GNN_node

# Get hyperparameters from config file
with open('config.json', 'r') as fh:
    params = json.load(fh)
    test = params['test'] # if only test but not train
    restart = params['restart'] # if restart training
    reload_dataset = params['reload_dataset'] # if reload already processed h_dataset

    if test:
        restart = True

    prediction = params['prediction'] # one of ['congestion', 'demand']
    model_type = params['model_type'] #this can be one of ["dehnn", "dehnn_att", "digcn", "digat"] "dehnn_att" might need large memory usage
    num_layer = params['num_layer'] #large number will cause OOM
    num_dim = params['num_dim'] #large number will cause OOM
    vn = params['vn'] #use virtual node or not
    trans = params['trans'] #use transformer or not
    aggr = params['aggr'] #use aggregation as one of ["add", "max"]
    device = params['device'] #use cuda or cpu
    learning_rate = params['learning_rate'] #rate of gradient descent
    num_epochs = params['num_epochs'] #length of training

if not reload_dataset:
    # open data files
    dataset = NetlistDataset(data_dir="../data/superblue", load_pd = True, load_pe = True, pl = True, processed = reload_dataset, load_indices=None)
    h_dataset = []
    for data in tqdm(dataset):
        # read in node and net features
        num_instances = data.node_features.shape[0]
        data.num_instances = num_instances
        data.edge_index_sink_to_net[1] = data.edge_index_sink_to_net[1] - num_instances
        data.edge_index_source_to_net[1] = data.edge_index_source_to_net[1] - num_instances
        
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
        h_data['net'].x = data.net_features.float()
        print(data.node_features.shape)

        # Create hypergraph structure
        edge_index = torch.concat([data.edge_index_sink_to_net, data.edge_index_source_to_net], dim=1)
        h_data['node', 'to', 'net'].edge_index, h_data['node', 'to', 'net'].edge_weight = gcn_norm(edge_index, add_self_loops=False)
        h_data['node', 'to', 'net'].edge_type = torch.concat([torch.zeros(data.edge_index_sink_to_net.shape[1]), torch.ones(data.edge_index_source_to_net.shape[1])]).bool()
        h_data['net', 'to', 'node'].edge_index, h_data['net', 'to', 'node'].edge_weight = gcn_norm(edge_index.flip(0), add_self_loops=False)
        
        h_data['design_name'] = data['design_name']
        h_data.num_instances = data.node_features.shape[0]
        variant_data_lst = []

        # Read congestion targets for prediction
        node_congestion = data.node_congestion.long()
        net_congestion = data.net_congestion.long()
        node_demand = data.node_demand.float()
        net_demand = data.net_demand.float()
        # node_demand = (node_demand - torch.mean(node_demand)) / torch.std(node_demand)
        # net_demand = (net_demand - torch.mean(net_demand))/ torch.std(net_demand)
        
        # Create and save processed data
        batch = data.batch
        num_vn = len(np.unique(batch))
        vn_node = torch.concat([global_mean_pool(h_data['node'].x, batch), 
                global_max_pool(h_data['node'].x, batch)], dim=1)

        variant_data_lst.append((node_demand, net_demand, node_congestion, net_congestion, batch, num_vn, vn_node, data.mask)) 
        h_data['variant_data_lst'] = variant_data_lst
        h_dataset.append(h_data)
        
    torch.save(h_dataset, "h_dataset.pt")
    
else:
    dataset = torch.load("h_dataset.pt")
    h_dataset = []
    for data in dataset:
        h_dataset.append(data)
    
sys.path.append("models/layers/")

# Create new GNN from given hyperparameters
h_data = h_dataset[0]
if restart:
    model = torch.load(f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_{prediction}.pt")
else:
    model = GNN_node(num_layer, num_dim, 1, 1, node_dim = h_data['node'].x.shape[1], net_dim = h_data['net'].x.shape[1], gnn_type=model_type, vn=vn, trans=trans, aggr=aggr, JK="Normal").to(device)

# Create loss function
criterion_node = nn.MSELoss()
criterion_net = nn.MSELoss()

# Training, validation, and test split
optimizer = optim.AdamW(model.parameters(), lr=learning_rate,  weight_decay=0.001)
load_data_indices = [idx for idx in range(len(h_dataset))]
all_train_indices = load_data_indices[:4] + load_data_indices[6:]
all_valid_indices = load_data_indices[4:5]
all_test_indices = load_data_indices[5:6]
best_total_val = None

# Create csv file to save training metrics
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
filepath = f"../results/baselines/{timestamp}__{num_epochs}_{num_layer}_{num_dim}_{vn}_{prediction}_baseline.csv"
with open(filepath, 'a') as f:
    f.write('Epoch,Train_RMSE,Valid_RMSE,Test_RMSE,')
    f.write('Train_MAE,Valid_MAE,Test_MAE,')
    f.write('Train_Pearson,Valid_Pearson,Test_Pearson,Time\n')

if not test:
    start_time = time.time()
    for epoch in range(num_epochs):
        np.random.shuffle(all_train_indices)
        loss_node_all = 0
        loss_net_all = 0
        val_loss_node_all = 0
        val_loss_net_all = 0
        test_loss_node_all = 0
        test_loss_net_all = 0

        train_mae = 0
        val_mae = 0
        test_mae = 0

        train_pearson = 0
        val_pearson = 0
        test_pearson = 0
        
        all_train_idx = 0
        for data_idx in tqdm(all_train_indices):
            data = h_dataset[data_idx]
            for inner_data_idx in range(len(data.variant_data_lst)):
                target_node_demand, target_net_demand, target_node_congestion, target_net_congestion, batch, num_vn, vn_node, mask = data.variant_data_lst[inner_data_idx]
                # target_net_demand = target_net_demand[mask]
                optimizer.zero_grad()
                data.batch = batch
                data.num_vn = num_vn
                data.vn = vn_node

                # Forward pass training data through model
                node_representation, net_representation = model(data, device)
                node_representation = torch.squeeze(node_representation).float()
                net_representation = torch.squeeze(net_representation).float()#[mask]

                loss_node = criterion_node(node_representation, target_node_demand.to(device))
                loss_net = criterion_net(net_representation, target_net_demand.to(device))
                
                # Backpropagate the loss
                loss = loss_net
                loss.backward()
                optimizer.step()   
    
                loss_node_all += loss_node.item()
                loss_net_all += loss_net.item()

                train_mae += mean_absolute_error(net_representation.detach().cpu().numpy(), target_net_demand.detach().cpu().numpy())
                coef, pval = pearsonr(net_representation.detach().cpu().numpy(), target_net_demand.detach().cpu().numpy())
                train_pearson += coef

                all_train_idx += 1

                
        # print('Training Loss Node: ', loss_node_all/all_train_idx, ', Net: ', loss_net_all/all_train_idx)
        print('Train RMSE: ', np.sqrt(loss_net_all/all_train_idx), ', Train MAE: ', train_mae/all_train_idx, ', Train Pearson: ', train_pearson/all_train_idx)
    
        all_valid_idx = 0
        for data_idx in tqdm(all_valid_indices):
            data = h_dataset[data_idx]
            for inner_data_idx in range(len(data.variant_data_lst)):
                target_node_demand, target_net_demand, target_node_congestion, target_net_congestion, batch, num_vn, vn_node, mask = data.variant_data_lst[inner_data_idx]
                # target_net_demand = target_net_demand[mask]
                data.batch = batch
                data.num_vn = num_vn
                data.vn = vn_node
                node_representation, net_representation = model(data, device)
                node_representation = torch.squeeze(node_representation)
                net_representation = torch.squeeze(net_representation)#[mask]
                
                # Check training results on validation set
                val_loss_node = criterion_node(node_representation, target_node_demand.to(device))
                val_loss_net = criterion_net(net_representation, target_net_demand.to(device))

                val_loss_node_all +=  val_loss_node.item()
                val_loss_net_all += val_loss_net.item()

                val_mae += mean_absolute_error(net_representation.detach().cpu().numpy(), target_net_demand.detach().cpu().numpy())
                coef, pval = pearsonr(net_representation.detach().cpu().numpy(), target_net_demand.detach().cpu().numpy())
                val_pearson += coef

                all_valid_idx += 1

        all_test_idx = 0
        for data_idx in tqdm(all_test_indices):
            data = h_dataset[data_idx]
            for inner_data_idx in range(len(data.variant_data_lst)):
                target_node_demand, target_net_demand, target_node_congestion, target_net_congestion, batch, num_vn, vn_node, mask = data.variant_data_lst[inner_data_idx]
                # target_net_demand = target_net_demand[mask]
                data.batch = batch
                data.num_vn = num_vn
                data.vn = vn_node
                node_representation, net_representation = model(data, device)
                node_representation = torch.squeeze(node_representation)
                net_representation = torch.squeeze(net_representation)#[mask]
                
                # Check training results on validation set           
                test_loss_node = criterion_node(node_representation, target_node_demand.to(device))
                test_loss_net = criterion_net(net_representation, target_net_demand.to(device))

                test_loss_node_all +=  test_loss_node.item()
                test_loss_net_all += test_loss_net.item()

                test_mae += mean_absolute_error(net_representation.detach().cpu().numpy(), target_net_demand.detach().cpu().numpy())
                coef, pval = pearsonr(net_representation.detach().cpu().numpy(), target_net_demand.detach().cpu().numpy())
                test_pearson += coef

                all_test_idx += 1

        # Save metrics for this epoch
        # print('Validation Loss Node: ', val_loss_node_all/all_valid_idx, ', Net: ',  val_loss_net_all/all_valid_idx)
        # print('Test Loss Node:       ', test_loss_node_all/all_test_idx, ', Net: ',  test_loss_net_all/all_test_idx)
        print('Valid RMSE: ', np.sqrt(val_loss_net_all/all_valid_idx), ', Valid MAE: ', val_mae/all_valid_idx, ', Valid Pearson: ', val_pearson/all_valid_idx)
        print('Test  RMSE: ', np.sqrt(test_loss_net_all/all_test_idx), ', Test  MAE: ', test_mae/all_test_idx, ', Test  Pearson: ', test_pearson/all_test_idx)
        
        print(f'Epoch {epoch+1}, {round(time.time() - start_time, 2)}s\n')
    
        if (best_total_val is None) or ((loss_net_all/all_train_idx) < best_total_val):
            best_total_val = loss_net_all/all_train_idx
            torch.save(model, f"{timestamp}_{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_{prediction}.pt")

        with open(filepath, 'a') as f:
            f.write(f'{epoch+1},{np.sqrt(loss_net_all/all_train_idx)},{np.sqrt(val_loss_net_all/all_valid_idx)},{np.sqrt(test_loss_net_all/all_test_idx)},')
            f.write(f'{train_mae/all_train_idx},{val_mae/all_valid_idx},{test_mae/all_test_idx},')
            f.write(f'{train_pearson/all_train_idx},{val_pearson/all_valid_idx},{test_pearson/all_test_idx},{round(time.time()-start_time, 2)}\n')
        
else:
    # Test model without training
    all_test_idx = 0
    test_loss_node_all = 0
    test_loss_net_all = 0
    for data_idx in tqdm(all_test_indices):
        data = h_dataset[data_idx]
        for inner_data_idx in range(len(data.variant_data_lst)):
            target_node_demand, target_net_demand, target_node_congestion, target_net_congestion, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
            data.batch = batch
            data.num_vn = num_vn
            data.vn = vn_node
            node_representation, net_representation = model(data, device)
            node_representation = torch.squeeze(node_representation)
            net_representation = torch.squeeze(net_representation)
            
            test_loss_node = criterion_node(node_representation, target_node_demand.to(device))
            test_loss_net = criterion_net(net_representation, target_net_demand.to(device))

            test_loss_node_all +=  test_loss_node.item()
            test_loss_net_all += test_loss_net.item()
            all_test_idx += 1

    print('Test Loss Node: ', test_loss_node_all/all_test_idx, ', Net: ',  test_loss_net_all/all_test_idx)
