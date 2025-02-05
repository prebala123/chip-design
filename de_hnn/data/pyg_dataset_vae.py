import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
import numpy as np

from utils import *


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        # Using sigmoid 
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class NetlistDataset(Dataset):
    def __init__(self, data_dir, load_pe=True, load_pd=True, num_eigen=10, pl=True, processed=False, load_indices=None, density=False):
        super().__init__()
        self.data_dir = data_dir
        self.data_lst = []

        all_files = np.array(os.listdir(data_dir))
        
        if load_indices is not None:
            load_indices = np.array(load_indices)
            all_files = all_files[load_indices]
            
        for design_fp in tqdm(all_files):
            data_load_fp = os.path.join(data_dir, design_fp, 'pyg_data.pkl')
            if processed and os.path.exists(data_load_fp):
                data = torch.load(data_load_fp)
            else:
                data_load_fp = os.path.join(data_dir, design_fp)

                # ----- Load node features and basic info -----
                file_name = os.path.join(data_load_fp, 'node_features.pkl')
                with open(file_name, 'rb') as f:
                    dictionary = pickle.load(f)
                self.design_name = dictionary['design']
                num_instances = dictionary['num_instances']
                num_nets = dictionary['num_nets']
                raw_instance_features = torch.Tensor(dictionary['instance_features'])
                pos_lst = raw_instance_features[:, :2]

                x_min = dictionary['x_min']
                x_max = dictionary['x_max']
                y_min = dictionary['y_min']
                y_max = dictionary['y_max'] 
                min_cell_width = dictionary['min_cell_width']
                max_cell_width = dictionary['max_cell_width']
                min_cell_height = dictionary['min_cell_height']
                max_cell_height = dictionary['max_cell_height']
                
                instance_features = raw_instance_features[:, 2:]
                # initialize net features with zeros (will later append extra features)
                net_features = torch.zeros(num_nets, instance_features.size(1))
                
                # ----- Load bipartite connectivity -----
                file_name = os.path.join(data_load_fp, 'bipartite.pkl')
                with open(file_name, 'rb') as f:
                    bip_dict = pickle.load(f)
                
                instance_idx = torch.Tensor(bip_dict['instance_idx']).unsqueeze(dim=1).long()
                net_idx = torch.Tensor(bip_dict['net_idx']) + num_instances
                net_idx = net_idx.unsqueeze(dim=1).long()
                local_net_idx = net_idx - num_instances  # local indices for nets
                
                edge_attr = torch.Tensor(bip_dict['edge_attr']).float().unsqueeze(dim=1).float()
                edge_index = torch.cat((instance_idx, net_idx), dim=1)
                edge_dir = bip_dict['edge_dir']
                v_drive_idx = [idx for idx in range(len(edge_dir)) if edge_dir[idx] == 1]
                v_sink_idx = [idx for idx in range(len(edge_dir)) if edge_dir[idx] == 0]
                edge_index_source_to_net = edge_index[v_drive_idx]
                edge_index_sink_to_net = edge_index[v_sink_idx]
                
                edge_index_source_to_net = torch.transpose(edge_index_source_to_net, 0, 1)
                edge_index_sink_to_net = torch.transpose(edge_index_sink_to_net, 0, 1)
                
                x = instance_features
                example = Data()
                example.__num_nodes__ = x.size(0)
                example.x = x

                # ----- Load degree information -----
                fn = os.path.join(data_load_fp, 'degree.pkl')
                with open(fn, "rb") as f:
                    d = pickle.load(f)
                example.edge_attr = edge_attr[:2]
                example.cell_degrees = torch.tensor(d['cell_degrees'])
                example.net_degrees = torch.tensor(d['net_degrees'])
                
                example.x = torch.cat([example.x, example.cell_degrees.unsqueeze(dim=1)], dim=1)
                example.x_net = example.net_degrees.unsqueeze(dim=1)

                # ----- Load partition information -----
                file_name = os.path.join(data_load_fp, 'metis_part_dict.pkl')
                with open(file_name, 'rb') as f:
                    part_dict = pickle.load(f)
                
                part_id_lst = []
                for idx in range(len(example.x)):
                    part_id_lst.append(part_dict[idx])
                part_id = torch.LongTensor(part_id_lst)
                example.part_id = part_id
                example.num_vn = len(torch.unique(part_id))
                top_part_id = torch.zeros(example.num_vn, dtype=torch.long)
                example.top_part_id = top_part_id

                # ----- (Partition Global Flow Feature omitted for brevity) -----

                # ----- Load net demand and targets -----
                file_name = os.path.join(data_load_fp, 'net_demand_capacity.pkl')
                with open(file_name, 'rb') as f:
                    net_demand_dictionary = pickle.load(f)
                net_demand = torch.Tensor(net_demand_dictionary['demand'])

                file_name = os.path.join(data_load_fp, 'targets.pkl')
                with open(file_name, 'rb') as f:
                    node_demand_dictionary = pickle.load(f)
                node_demand = torch.Tensor(node_demand_dictionary['demand'])
                
                fn = os.path.join(data_load_fp, 'net_hpwl.pkl')
                with open(fn, 'rb') as f:
                    d_hpwl = pickle.load(f)
                net_hpwl = torch.Tensor(d_hpwl['hpwl']).float()

                # ----- Spectral and Laplacian Features -----
                if load_pe:
                    file_name = os.path.join(data_load_fp, 'eigen.10.pkl')
                    with open(file_name, 'rb') as f:
                        eigen_dictionary = pickle.load(f)
                    evects = torch.Tensor(eigen_dictionary['evects'])
                    example.x = torch.cat([example.x, evects[:example.x.shape[0]]], dim=1)
                    example.x_net = torch.cat([example.x_net, evects[example.x.shape[0]:]], dim=1)
                    if 'evals' in eigen_dictionary:
                        evals = torch.Tensor(eigen_dictionary['evals'])
                        spectral_gap = evals[1] - evals[0]
                    else:
                        spectral_gap = torch.tensor(0.0)
                    spectral_gap_node = spectral_gap.repeat(example.x.shape[0], 1)
                    spectral_gap_net = spectral_gap.repeat(example.x_net.shape[0], 1)
                    example.x = torch.cat([example.x, spectral_gap_node], dim=1)
                    example.x_net = torch.cat([example.x_net, spectral_gap_net], dim=1)

                # ----- Load Persistent Diagram / Neighbor Features -----
                if load_pd:
                    file_name = os.path.join(data_load_fp, 'node_neighbor_features.pkl')
                    with open(file_name, 'rb') as f:
                        pd_dict = pickle.load(f)
                    pd = torch.Tensor(pd_dict['pd'])
                    neighbor_list = torch.Tensor(pd_dict['neighbor'])
                    assert pd.size(0) == num_instances
                    assert neighbor_list.size(0) == num_instances
                    example.x = torch.cat([example.x, pd, neighbor_list], dim=1)

                # --- Previously implemented Forman curvature block could be here) ---

                # ===== NEW: Variational Autoencoder on node features =====
                # Train a simple VAE on the current node features (example.x)
                # and then append the latent representation to the original features.
                input_dim = example.x.shape[1]
                hidden_dim = 32   # you may adjust these dimensions
                latent_dim = 8
                vae = VAE(input_dim, hidden_dim, latent_dim)
                optimizer = optim.Adam(vae.parameters(), lr=1e-3)
                num_epochs = 50  # number of training epochs; adjust as needed
                x_data = example.x  # shape: (num_instances, input_dim)
                # Training loop (on the node features of the design)
                for epoch in range(num_epochs):
                    optimizer.zero_grad()
                    z, mu, logvar = vae(x_data)
                    recon_x = vae.decode(z)
                    recon_loss = nn.functional.mse_loss(recon_x, x_data)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_loss
                    loss.backward()
                    optimizer.step()
                # After training, obtain the latent representation and append it.
                with torch.no_grad():
                    z, _, _ = vae(x_data)
                # Append the latent code (z) as additional features to example.x
                example.x = torch.cat([example.x, z], dim=1)
                # ===== End VAE block =====

                # ----- Build final Data object -----
                data = Data(
                    node_features=example.x, 
                    net_features=example.x_net, 
                    edge_index_sink_to_net=edge_index_sink_to_net, 
                    edge_index_source_to_net=edge_index_source_to_net, 
                    node_demand=node_demand, 
                    net_demand=net_demand,
                    net_hpwl=net_hpwl,
                    batch=example.part_id,
                    num_vn=example.num_vn,
                    pos_lst=pos_lst
                )
                
                data_save_fp = os.path.join(data_load_fp, 'pyg_data.pkl')
                torch.save(data, data_save_fp)
                    
            data['design_name'] = design_fp
            self.data_lst.append(data)

    def len(self):
        return len(self.data_lst)

    def get(self, idx):
        return self.data_lst[idx]
