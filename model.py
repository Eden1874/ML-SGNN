import torch

from torchtools import *
from collections import OrderedDict


class EmbeddingCredit(nn.Module):
    def __init__(self, emb_size):
        super(EmbeddingCredit, self).__init__()

        self.emb_size = emb_size
        self.conv_1 = nn.Sequential(nn.Linear(198, 128),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.2))
        # self.se = nn.Sequential(nn.Linear(128, 16, bias=False),
        #                         nn.ReLU(inplace=True),
        #                         nn.Linear(16, 128, bias=False),
        #                         nn.Sigmoid())
        self.conv_2 = nn.Sequential(nn.Linear(128, emb_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.2))

    def forward(self, input_data):
        input_data = input_data.view(tt.arg.meta_batch_size * (tt.arg.num_shots * 2 + 2), 198)
        output_data = self.conv_1(input_data)
        # output_data = output_data * self.se(output_data)
        output_data = self.conv_2(output_data)
        return output_data.view(tt.arg.meta_batch_size, tt.arg.num_shots * 2 + 2, self.emb_size)


class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 reduction=16):
        super(NodeUpdateNetwork, self).__init__()

    def forward(self, node_feat, edge_feat):
        # get size
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 2, 1, 1).to(tt.arg.device)
        edge_feat = torch.split(edge_feat * diag_mask, 1, 1)

        node_weight = F.relu(edge_feat[0] - edge_feat[1])

        node_weight = F.normalize(node_weight, p=1, dim=-1)
        node_weight = F.normalize(node_weight + torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 1, 1, 1).to(tt.arg.device), p=1, dim=-1)
        node_feat = torch.bmm(node_weight.squeeze(1), node_feat)

        return node_feat


class EdgeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 separate_dissimilarity=True ):
        super(EdgeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.separate_dissimilarity = separate_dissimilarity

        # layers
        self.sim_network = nn.Sequential(nn.Conv2d(in_channels=self.in_features,
                                                   out_channels=1,
                                                   kernel_size=1))

        if self.separate_dissimilarity:
            self.dsim_network = nn.Sequential(nn.Conv2d(in_channels=self.in_features,
                                                        out_channels=1,
                                                        kernel_size=1))

    def forward(self, node_feat, edge_feat):
        # compute abs(x_i, x_j)
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)

        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = F.sigmoid(self.sim_network(x_ij))

        if self.separate_dissimilarity:
            dsim_val = F.sigmoid(self.dsim_network(x_ij))
        else:
            dsim_val = 1.0 - sim_val

        diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 2, 1, 1).to(tt.arg.device)
        edge_feat = edge_feat * diag_mask
        merge_sum = torch.sum(edge_feat, -1, True)
        # set diagonal as zero and normalize
        edge_feat = F.normalize(torch.cat([sim_val, dsim_val], 1) * edge_feat, p=1, dim=-1) * merge_sum
        
        force_edge_feat = torch.cat((torch.eye(node_feat.size(1)).unsqueeze(0), torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0)), 0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).to(tt.arg.device)
        edge_feat = edge_feat + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1).repeat(1, 2, 1, 1)

        return edge_feat


class GraphNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 node_features,
                 edge_features,
                 num_layers,
                 dropout=0.0):
        super(GraphNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout

        # for each layer
        for l in range(self.num_layers):
            # set edge to node
            edge2node_net = NodeUpdateNetwork(in_features=self.in_features)

            # set node to edge
            node2edge_net = EdgeUpdateNetwork(in_features=self.edge_features)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)

    # forward
    def forward(self, node_feat, edge_feat):
        # for each layer
        edge_feat_list = []
        for l in range(self.num_layers):
            # (1) edge to node
            node_feat = self._modules['edge2node_net{}'.format(l)](node_feat, edge_feat)

            # (2) node to edge
            edge_feat = self._modules['node2edge_net{}'.format(l)](node_feat, edge_feat)

            # save edge feature
            # edge_feat_list.append(edge_feat)

        return edge_feat

