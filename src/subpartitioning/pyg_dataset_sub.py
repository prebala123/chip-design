import os
import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
import numpy as np

from utils import *
from gen_subpartitions import *

class NetlistDataset(Dataset):
    def __init__(self, data_dir, load_pe = True, load_pd = True, num_eigen = 10, pl = True, processed = False, load_indices = None, density = False):
        super().__init__()
        self.data_dir = data_dir
        self.data_lst = []

        all_files = np.array(os.listdir(data_dir))
        
        if load_indices is not None:
            load_indices = np.array(load_indices)
            all_files = all_files[load_indices]
            
        for design_fp in tqdm(all_files):
            if design_fp[:9] != 'superblue':
                print(design_fp + ' skipped')
                continue
            #print(design_fp)
            data_load_fp = os.path.join(data_dir, design_fp, 'subpartitions_pyg_data')
            if processed and os.path.exists(data_load_fp):
                for subpartition in os.listdir(data_load_fp):
                    data = torch.load(os.path.join(data_load_fp, subpartition))
                    label = subpartition[:subpartition.index('_')]
                    data['design_name'] = design_fp + f'_{label}'
                    # print(data['design_name'])
                    self.data_lst.append(data)
            else:
                data_load_fp = os.path.join(data_dir, design_fp)

                file_name = data_load_fp + '/' + 'node_features.pkl'
                f = open(file_name, 'rb')
                node_features_dictionary = pickle.load(f)
                f.close()        
                self.design_name = node_features_dictionary['design']
                num_instances = node_features_dictionary['num_instances']
                num_nets = node_features_dictionary['num_nets']
                raw_instance_features = torch.Tensor(node_features_dictionary['instance_features'])
                pos_lst = raw_instance_features[:, :2]
                
                instance_features = raw_instance_features[:, 2:]
                net_features = torch.zeros(num_nets, instance_features.size(1))
                
                file_name = data_load_fp + '/' + 'bipartite.pkl'
                f = open(file_name, 'rb')
                bipartite_dictionary = pickle.load(f)
                f.close()
        
                instance_idx = torch.Tensor(bipartite_dictionary['instance_idx']).unsqueeze(dim = 1).long()
                net_idx = torch.Tensor(bipartite_dictionary['net_idx']) + num_instances
                net_idx = net_idx.unsqueeze(dim = 1).long()
                
                edge_index = torch.cat((instance_idx, net_idx), dim = 1)
                edge_dir = bipartite_dictionary['edge_dir']
                
                x = instance_features
                
                example = Data()
                example.__num_nodes__ = x.size(0)
                example.x = x # instance_features (,4)

                file_name = data_load_fp + '/' + 'metis_part_dict.pkl'
                f = open(file_name, 'rb')
                part_dict = pickle.load(f)
                f.close()

                part_id_lst = []

                for idx in range(len(example.x)):
                    part_id_lst.append(part_dict[idx])

                part_id = torch.LongTensor(part_id_lst)

                example.num_vn = len(torch.unique(part_id))

                top_part_id = torch.Tensor([0 for idx in range(example.num_vn)]).long()

                example.num_top_vn = len(torch.unique(top_part_id))

                example.part_id = part_id
                example.top_part_id = top_part_id

                file_name = data_load_fp + '/' + 'net_demand_capacity.pkl'
                f = open(file_name, 'rb')
                net_demand_dictionary = pickle.load(f)
                f.close()

                net_demand = torch.Tensor(net_demand_dictionary['demand'])
                example.x_net = torch.Tensor(net_demand_dictionary['capacity']).unsqueeze(dim = 1) # capacity (,1)

                file_name = data_load_fp + '/' + 'targets.pkl'
                f = open(file_name, 'rb')
                node_demand_dictionary = pickle.load(f)
                f.close()

                node_demand = torch.Tensor(node_demand_dictionary['demand'])
                node_capacity = torch.Tensor(node_demand_dictionary['capacity'])
                example.x = torch.cat([node_capacity.unsqueeze(dim=1), example.x], dim = 1) # capacity (,1) + instance_features (,4)
                
                fn = data_load_fp + '/' + 'net_hpwl.pkl'
                f = open(fn, "rb")
                d_hpwl = pickle.load(f)
                f.close()
                net_hpwl = torch.Tensor(d_hpwl['hpwl']).float()

                if os.path.exists(os.path.join(data_load_fp, "subpartitions100.pkl")):
                    subpartitions_fp = os.path.join(data_load_fp, 'subpartitions100.pkl')
                    f = open(subpartitions_fp, 'rb')
                    subpartitions_dict = pickle.load(f)
                    f.close()
                else:
                    adjacency = build_adjacency(bipartite_dictionary, node_features_dictionary)
                    # Compute the hierarchical subpartitions (expected to be 100 subpartitions).
                    subpartitions_dict, _ = compute_hierarchical_subpartitions(adjacency, ufactor=600, nparts=10)
                    # Save the subpartitions dictionary as 'subpartitions100.pkl' in the folder.
                    output_fp = os.path.join(data_load_fp, "subpartitions100.pkl")
                    with open(output_fp, 'wb') as f:
                        pickle.dump(subpartitions_dict, f)
                    print(f"Saved subpartitions to {output_fp}")

                for label, nodes in subpartitions_dict.items():
                    nodes = sorted(nodes)
                    instance_indices = [node for node in nodes if node < num_instances]
                    net_indices = [(node - num_instances) for node in nodes if node >= num_instances]

                    example.x_sub = example.x[instance_indices] # capacity (,1) + instance_features (,4)

                    subpartition_degree_fp = os.path.join(data_load_fp, 'subpartitions_degree', f'{label}_degree.pkl')
                    with open(subpartition_degree_fp, 'rb') as f:
                        subpartition_degree_dict = pickle.load(f)

                    example.cell_degrees = torch.tensor(subpartition_degree_dict['cell_degrees'])
                    example.net_degrees = torch.tensor(subpartition_degree_dict['net_degrees'])
                    
                    example.x_sub = torch.cat([example.x_sub, example.cell_degrees.unsqueeze(dim = 1)], dim = 1) # capacity (,1) + instance_features (,4) + degrees(,1)
                    example.x_net_sub = torch.cat([example.x_net[net_indices], example.net_degrees.unsqueeze(dim = 1)], dim = 1) # capacity (,1) + degrees (,1)

                    subpartition_eigen_fp = os.path.join(data_load_fp, 'subpartitions_eigen', f'{label}_eigen.10.pkl')
                    with open(subpartition_eigen_fp, 'rb') as f:
                        subpartition_eigen_dict = pickle.load(f)

                    evects = torch.Tensor(subpartition_eigen_dict['evects'])
                    example.x_sub = torch.cat([example.x_sub, evects[:example.x_sub.shape[0]]], dim = 1) # capacity (,1) + instance_features (,4) + degrees(,1) + evects(,10)
                    example.x_net_sub = torch.cat([example.x_net_sub, evects[example.x_sub.shape[0]:]], dim = 1) # capacity (,1) + degrees (,1) + evects(,10)

                    nodes_dict = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes)}

                    v_drive_idx = [idx for idx in range(len(edge_dir)) if edge_dir[idx] == 1 and edge_index[idx,0].item() in nodes_dict and edge_index[idx,1].item() in nodes_dict]
                    v_sink_idx = [idx for idx in range(len(edge_dir)) if edge_dir[idx] == 0 and edge_index[idx,0].item() in nodes_dict and edge_index[idx,1].item() in nodes_dict] 
                    edge_index_source_to_net = edge_index[v_drive_idx]
                    edge_index_sink_to_net = edge_index[v_sink_idx]

                    max_old = max(nodes_dict.keys())
                    mapping_tensor = torch.zeros(max_old + 1, dtype=torch.long)
                    for old_idx, new_idx in nodes_dict.items():
                        mapping_tensor[old_idx] = new_idx
                    
                    edge_index_source_to_net = torch.transpose(edge_index_source_to_net, 0, 1)
                    edge_index_sink_to_net = torch.transpose(edge_index_sink_to_net, 0, 1)

                    edge_index_source_to_net = mapping_tensor[edge_index_source_to_net]
                    edge_index_sink_to_net = mapping_tensor[edge_index_sink_to_net]

                    data = Data(
                            node_features = example.x_sub, # capacity (,1) + instance_features (,4) + degrees(,1) + evects(,10)
                            net_features = example.x_net_sub, # capacity (,1) + degrees (,1) + evects(,10)
                            edge_index_sink_to_net = edge_index_sink_to_net, 
                            edge_index_source_to_net = edge_index_source_to_net, 
                            node_demand = node_demand[instance_indices], 
                            net_demand = net_demand[net_indices],
                            net_hpwl = net_hpwl[net_indices],
                            batch = example.part_id, # ignored later
                            num_vn = example.num_vn, # ignored later
                            pos_lst = pos_lst[instance_indices]
                        )
                    
                    directory = os.path.join(data_load_fp, 'subpartitions_pyg_data')
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    data_save_fp = os.path.join(directory, f'{label}_pyg_data.pkl')
                    torch.save(data, data_save_fp)
                    

                    data['design_name'] = design_fp + f'_{label}'
                    print(data['design_name'] + ' pyg_data object saved')
                    self.data_lst.append(data)

            
    def len(self):
        return len(self.data_lst)

    def get(self, idx):
        return self.data_lst[idx]
