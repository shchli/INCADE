import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layer import *
from rgcn.layers import UnionRGCNLayer

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.device = args.device

    @staticmethod
    def format_metrics(metrics, split):
        return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])

    @staticmethod
    def has_improved(m1, m2):
        return (m1["Mean Rank"] > m2["Mean Rank"]) or (m1["Mean Reciprocal Rank"] < m2["Mean Reciprocal Rank"])

    @staticmethod
    def init_metric_dict():
        return {"Hits@100": -1, "Hits@10": -1, "Hits@3": -1, "Hits@1": -1,
                "MR": 100000, "MRR": -1}


class Mutan(BaseModel):
    def __init__(self, args):
        super(Mutan, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        self.dim = args.dim
        self.Mutan = MutanLayer(args.dim, 5)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        pred = self.Mutan(e_embed, r_embed)
        pred = torch.mm(pred, self.entity_embeddings.weight.transpose(1, 0))
        pred = torch.sigmoid(pred)
        return pred

    def loss_func(self, output, target):
        return self.bceloss(output, target)


class TuckER(BaseModel):
    def __init__(self, args):
        super(TuckER, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.r_dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        if args.pre_trained:
            self.entity_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float(), freeze=False)
            self.relation_embeddings = nn.Embedding.from_pretrained(torch.cat((
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float(),
                -1 * torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()), dim=0), freeze=False)
        self.dim = args.dim
        self.TuckER = TuckERLayer(args.dim, args.r_dim)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def forward(self, batch_inputs, lookup=None):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        pred = self.TuckER(e_embed, r_embed)
        if lookup is None:
            pred = torch.mm(pred, self.entity_embeddings.weight.transpose(1, 0))
        else:
            pred = torch.bmm(pred.unsqueeze(1), self.entity_embeddings.weight[lookup].transpose(1, 2)).squeeze(1)
        pred = torch.sigmoid(pred)
        return pred

    def loss_func(self, output, target):
        return self.bceloss(output, target)


class ConvE(BaseModel):
    def __init__(self, args):
        super(ConvE, self).__init__(args)
        self.entity_embeddings = nn.Embedding(len(args.entity2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(2 * len(args.relation2id), args.dim, padding_idx=None)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        self.dim = args.dim
        self.k_w = args.k_w
        self.k_h = args.k_h
        self.ConvE = ConvELayer(args.dim, args.out_channels, args.kernel_size, args.k_h, args.k_w)
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()

    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        e_embed = self.entity_embeddings(head)
        r_embed = self.relation_embeddings(relation)
        e_embed = e_embed.view(-1, 1, self.dim)
        r_embed = r_embed.view(-1, 1, self.dim)
        embed = torch.cat([e_embed, r_embed], dim=1)
        embed = torch.transpose(embed, 2, 1).reshape((-1, 1, 2 * self.k_w, self.k_h))

        pred = self.ConvE(embed)
        pred = torch.mm(pred, self.entity_embeddings.weight.transpose(1, 0))
        pred += self.bias.expand_as(pred)
        pred = torch.sigmoid(pred)
        return pred

    def loss_func(self, output, target):
        return self.bceloss(output, target)


class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, encoder_name="", opn="sub", rel_emb=None, use_cuda=False, analysis=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.skip_connect = skip_connect
        self.self_loop = self_loop
        self.encoder_name = encoder_name
        self.use_cuda = use_cuda
        self.run_analysis = analysis
        self.skip_connect = skip_connect
        print("use layer :{}".format(encoder_name))
        self.rel_emb = rel_emb
        self.opn = opn
        # create rgcn layers
        self.build_model()
        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):

            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        print("h before GCN message passing")
        print(g.ndata['h'])
        print("h behind GCN message passing")
        for layer in self.layers:
            layer(g)
        print(g.ndata['h'])
        return g.ndata.pop('h')


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "convgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "convgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')