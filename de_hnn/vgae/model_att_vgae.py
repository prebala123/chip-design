import torch
import math
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Sequential as Seq, Linear, LeakyReLU
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor, torch_sparse
from torch_geometric.utils import add_remaining_self_loops, negative_sampling, remove_self_loops
from torch_geometric.utils import dropout_edge

import sys
sys.path.append("./layers/")
from dehnn_layers import HyperConvLayer
from dgnn_layers import DiGraphConvLayer


class GNN_node(nn.Module):
    """
    A GNN for node and net representation that optionally includes a variational autoencoder branch.
    When use_vgae=True, the model returns:
      (node_supervised, net_supervised, rec_loss, kl_loss)
    Otherwise, it returns (node_supervised, net_supervised).
    """
    def __init__(self, num_layer, emb_dim, out_node_dim, out_net_dim, JK="concat", residual=True, 
                 gnn_type='dehnn', norm_type="layer", aggr="add", 
                 node_dim=None, net_dim=None, num_nodes=None, vn=True, trans=False, device='cpu',
                 use_vgae=True): # set use_vgae to True
        super(GNN_node, self).__init__()
        self.device = device
        self.num_layer = num_layer
        self.JK = JK
        self.residual = residual
        self.node_dim = node_dim
        self.net_dim = net_dim
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.out_node_dim = out_node_dim
        self.out_net_dim = out_net_dim
        self.gnn_type = gnn_type
        self.vn = vn
        self.trans = trans
        self.use_vgae = use_vgae  # set to True above

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.net_encoder = nn.Sequential(
            nn.Linear(net_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
                
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if self.vn:
            if self.trans:
                self.transformer_virtualnode_list = nn.ModuleList()
            
            self.virtualnode_encoder = nn.Sequential(
                nn.Linear(node_dim*2, emb_dim*2),
                nn.LeakyReLU(),
                nn.Linear(emb_dim*2, emb_dim)
            )

            self.mlp_virtualnode_list = nn.ModuleList()
            self.back_virtualnode_list = nn.ModuleList()

            for layer in range(num_layer):
                self.mlp_virtualnode_list.append(
                    nn.Sequential(
                        nn.Linear(emb_dim*2, emb_dim), 
                        nn.LeakyReLU(),
                        nn.Linear(emb_dim, emb_dim)
                    )
                )
                self.back_virtualnode_list.append(
                    nn.Sequential(
                        nn.Linear(emb_dim*2, emb_dim), 
                        nn.LeakyReLU(),
                        nn.Linear(emb_dim, emb_dim)
                    )
                )
                if self.trans:
                    self.transformer_virtualnode_list.append(
                        nn.TransformerEncoderLayer(d_model=emb_dim*2, nhead=8, dim_feedforward=512)
                    )

        for layer in range(num_layer):
            if gnn_type == 'digcn':
                self.convs.append(DiGraphConvLayer(emb_dim, emb_dim, aggr=aggr))
            elif gnn_type == 'digat':
                self.convs.append(DiGraphConvLayer(emb_dim, emb_dim, aggr=aggr, att=True))
            elif gnn_type == 'dehnn':
                self.convs.append(HyperConvLayer(emb_dim, emb_dim, aggr=aggr))
            elif gnn_type == 'dehnn_att':
                self.convs.append(HyperConvLayer(emb_dim, emb_dim, aggr=aggr, att=True))
            
            if norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(emb_dim))
            elif norm_type == "layer":
                self.norms.append(nn.LayerNorm(emb_dim))
            else:
                raise NotImplementedError("Norm type not implemented")

        self.fc1_node = nn.Linear(emb_dim, 256)
        self.fc2_node = nn.Linear(256, self.out_node_dim)

        self.fc1_net = nn.Linear(emb_dim, 256)
        self.fc2_net = nn.Linear(256, self.out_net_dim)

        # If using VGAE, add layers to compute variational parameters
        if self.use_vgae:
            self.node_mu = nn.Linear(emb_dim, emb_dim)
            self.node_logvar = nn.Linear(emb_dim, emb_dim)
            self.net_mu = nn.Linear(emb_dim, emb_dim)
            self.net_logvar = nn.Linear(emb_dim, emb_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_node, z_net, edge_index):
        # Inner product decoder for bipartite graph
        src, dst = edge_index
        # For each edge from a node (src) to a net (dst), compute dot product
        return torch.sigmoid((z_node[src] * z_net[dst]).sum(dim=1))

    def forward(self, data, device):
        h_inst = data['node'].x.to(device)
        h_net = data['net'].x.to(device)
        
        edge_index_node_to_net, edge_weight_node_to_net = data['node', 'to', 'net'].edge_index, data['node', 'to', 'net'].edge_weight
        edge_index_net_to_node, edge_weight_net_to_node = data['net', 'to', 'node'].edge_index, data['net', 'to', 'node'].edge_weight

        # Drop edge dropout
        edge_index_node_to_net, edge_mask = dropout_edge(edge_index_node_to_net, p=0.2)
        edge_index_net_to_node = edge_index_net_to_node[:, edge_mask]
        edge_weight_node_to_net = edge_weight_node_to_net[edge_mask]
        edge_weight_net_to_node = edge_weight_net_to_node[edge_mask]
        edge_type_node_to_net = data['node', 'to', 'net'].edge_type[edge_mask]
        
        num_instances = data.num_instances
        
        h_inst = self.node_encoder(h_inst) 
        h_net = self.net_encoder(h_net) 
        
        if self.vn:
            batch = data.batch.to(device)
            virtualnode_embedding = self.virtualnode_encoder(data.vn.to(device))
            
        # Message passing layers
        for layer in range(self.num_layer):
            if self.vn:
                h_inst = self.back_virtualnode_list[layer](torch.cat([h_inst, virtualnode_embedding[batch]], dim=1)) + h_inst
               
            h_inst, h_net = self.convs[layer](h_inst, h_net, edge_index_node_to_net, edge_weight_node_to_net,
                                              edge_type_node_to_net, edge_index_net_to_node, edge_weight_net_to_node, device)
            h_inst = self.norms[layer](h_inst)
            h_net = self.norms[layer](h_net)
            h_inst = F.leaky_relu(h_inst)
            h_net = F.leaky_relu(h_net)
            
            if (layer < self.num_layer - 1) and self.vn:
                virtualnode_embedding_temp = torch.cat([global_mean_pool(h_inst, batch), 
                                                        global_max_pool(h_inst, batch)], dim=1)
                if self.trans:
                    virtualnode_embedding_temp = self.transformer_virtualnode_list[layer](virtualnode_embedding_temp) 
                    virtualnode_embedding = self.mlp_virtualnode_list[layer](virtualnode_embedding_temp) + virtualnode_embedding
                else:
                    virtualnode_embedding = self.mlp_virtualnode_list[layer](virtualnode_embedding_temp) + virtualnode_embedding
            
        # For supervised prediction: use an MLP on top of h_inst and h_net
        h_inst_sup = self.fc2_node(F.leaky_relu(self.fc1_node(h_inst)))
        h_net_sup = self.fc2_net(F.leaky_relu(self.fc1_net(h_net)))
        
        if self.use_vgae:
            # Compute variational parameters for nodes
            node_mu = self.node_mu(h_inst)
            node_logvar = self.node_logvar(h_inst)
            z_node = self.reparameterize(node_mu, node_logvar)
            # Compute variational parameters for nets
            net_mu = self.net_mu(h_net)
            net_logvar = self.net_logvar(h_net)
            z_net = self.reparameterize(net_mu, net_logvar)
            
            # Decode the bipartite edges from node to net
            pos_edge_index = data['node', 'to', 'net'].edge_index  # positive (observed) edges
            pos_pred = self.decode(z_node, z_net, pos_edge_index)
            # Generate negative edges for reconstruction loss (using torch_geometric's negative_sampling)
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index, num_nodes=(h_inst.size(0), h_net.size(0)),
                num_neg_samples=pos_edge_index.size(1)
            )
            neg_pred = self.decode(z_node, z_net, neg_edge_index)
            
            # Labels: 1 for positive edges, 0 for negatives
            pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
            neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
            rec_loss = pos_loss + neg_loss
            
            # Compute KL divergence losses for nodes and nets
            kl_node = -0.5 * torch.mean(torch.sum(1 + node_logvar - node_mu.pow(2) - node_logvar.exp(), dim=1))
            kl_net = -0.5 * torch.mean(torch.sum(1 + net_logvar - net_mu.pow(2) - net_logvar.exp(), dim=1))
            kl_loss = kl_node + kl_net
            
            return h_inst_sup, h_net_sup, rec_loss, kl_loss
        else:
            return h_inst_sup, h_net_sup