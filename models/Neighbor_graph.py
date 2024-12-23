import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import dgl
import math
class NodeModel(torch.nn.Module):
    def edge2node(self, node_num, x, rel_type):
        mask = rel_type.squeeze()
        x = x + x * (mask.unsqueeze(0))
        # rel = torch.tensor(np.ones(shape=(node_num,x.size()[0]))).cuda()
        rel = torch.ones((node_num, x.size(0)), device=x.device)
        incoming = torch.matmul(rel.to(torch.float32), x)
        return incoming / incoming.size(1)
    def __init__(self,node_h,edge_h,gnn_h,group_num, channel_dim=120, time_reduce_size=1):
        super(NodeModel, self).__init__()
        powernum = int(math.log2(group_num))
        channel_dim = (2 ** (powernum-1))*(group_num-1)
        self.node_mlp_1 = Seq(Lin(node_h+edge_h,gnn_h), ReLU(inplace=True))
        self.node_mlp_2 = Seq(Lin(node_h+gnn_h,gnn_h), ReLU(inplace=True))
        self.conv3 = nn.Conv1d(channel_dim * time_reduce_size * 2, channel_dim * time_reduce_size * 2, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(channel_dim * time_reduce_size * 2)
        self.conv4 = nn.Conv1d(channel_dim * time_reduce_size * 2, 1, kernel_size=1, stride=1)
        self.conv5 = nn.Conv1d(channel_dim * time_reduce_size * 2, channel_dim * time_reduce_size * 2, kernel_size=1, stride=1)
    def forward(self, x, edge_index, edge_attr):
        edge = edge_attr
        node_num = x.size()[0]
        edge = F.relu(self.conv3(edge))
        x = self.conv4(edge)
        rel_type = torch.sigmoid(x)
        s_input_2 = self.edge2node(node_num, edge, rel_type)
        return s_input_2
    
class Community_graph(nn.Module):

    def __init__(self, x_em=0, edge_h=0, gnn_h=0, gnn_layer=0, city_num=0, group_num=0,
                 device=0,groupt=None, w_init="rand"):
        super().__init__()
        self.gnn_layer = gnn_layer
        self.city_num = city_num
        self.group_num = group_num
        self.x_em = x_em
        self.device = device
        self.w_init = w_init
        self.device = device
        self.gnn_h = gnn_h
        if self.w_init == 'rand':
            if groupt != None and groupt.shape[0]==city_num:
                self.w = Parameter(groupt.to(device, non_blocking=True), requires_grad=True)
            else:
                self.w = Parameter(torch.randn(city_num, group_num).to(device, non_blocking=True), requires_grad=True)
        elif self.w_init == 'group':
            self.w = Parameter(self.new_w, requires_grad=True)
        self.lin1 = nn.Linear(x_em,x_em, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(x_em), requires_grad=True)
        self.lin2 = nn.Linear(x_em,x_em, bias=False)
        self.bias2 = nn.Parameter(torch.zeros(x_em), requires_grad=True)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.attention = nn.Linear(x_em, 1)
        self.edge_inf = Seq(Lin(x_em * 2, edge_h), ReLU(inplace=True))
        self.group_gnn = nn.ModuleList([NodeModel(x_em, edge_h, gnn_h,group_num).to(device)])
        for i in range(self.gnn_layer - 1):
            self.group_gnn.append(NodeModel(gnn_h, edge_h, gnn_h,group_num).to(device))
        self.lin_x = Lin(x_em, edge_h, bias=False)

    def batchInput(self, x, edge_w, edge_index):
        sta_num = x.shape[1]
        x = x.reshape(-1, x.shape[-1])
        edge_w = edge_w.reshape(-1, edge_w.shape[-1])
        batch_offsets = torch.arange(edge_index.size(0), device=edge_index.device) * sta_num
        edge_index = edge_index + batch_offsets.view(-1, 1, 1)
        edge_index = edge_index.transpose(0, 1)
        edge_index = edge_index.reshape(2, -1)
        return x, edge_w, edge_index
    def forward(self, x,head=None):
        x_0 = self.lin_x(x)
        x = x.unsqueeze(0).unsqueeze(2) 
        x = x.reshape(-1, x.shape[2], x.shape[3])  
        self.city_num = x.shape[0] 
        x = x.reshape(-1, self.city_num, 1, x.shape[-1])
        x = torch.max(x, dim=-2).values
        if head is not None:
            current_w = self.w[head] 
        else:
            current_w = self.w
        w0 = F.softmax(current_w,dim=1)
        w1 = w0.transpose(0, 1)
        w1 = w1.unsqueeze(dim=0)
        w1 = w1.repeat_interleave(x.size(0), dim=0)
        g_x = torch.bmm(w1, x)
        i_idx, j_idx = torch.meshgrid(torch.arange(self.group_num, device=self.device),torch.arange(self.group_num, device=self.device), indexing='ij')
        mask = i_idx != j_idx
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        g_edge_input = torch.cat([g_x[:, i_idx], g_x[:, j_idx]], dim=-1)
        g_edge_w = self.edge_inf(g_edge_input) 
        g_edge_w = torch.sigmoid(g_edge_w)
        g_edge_index = torch.stack([i_idx, j_idx], dim=0)
        
        g_edge_w = g_edge_w.transpose(0, 1)
        g_edge_index = g_edge_index.unsqueeze(dim=0)
        g_edge_index = g_edge_index.transpose(1, 2)
        g_x, g_edge_w, g_edge_index = self.batchInput(g_x, g_edge_w, g_edge_index)
        for i in range(self.gnn_layer):
            g_x = self.group_gnn[i](g_x, g_edge_index, g_edge_w)  
        w2 = w0
        new_x = torch.mm(w2, g_x)
        
        return new_x.squeeze()+torch.relu(x_0)