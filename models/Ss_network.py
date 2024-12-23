import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layer import *
from .model import BaseModel,RGCNCell
from .Community_graph import Community_graph









class ExpertEncoder(nn.Module):

    def __init__(self, n_exps, layers, dropout=0.0,x_em=None,edge_h=None,gnn_h=None,gnn_layer=2,city_num=None,group_num=None,device=None,groupt=None, noise=True):
        super(ExpertEncoder, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise
        self.experts = nn.ModuleList([Community_graph(x_em=x_em,edge_h=edge_h,gnn_h=gnn_h,gnn_layer=gnn_layer,city_num=city_num,group_num=group_num,device=device,groupt=groupt) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, r=None, train=None, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        if r is not None:
            gates = F.softmax(logits / torch.sigmoid(r), dim=-1)
        else:
            gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x, r=None,head=None):
        gates = self.noisy_top_k_gating(x, r, self.training)
        if head != None:
            expert_outputs = [self.experts[i](x,head).unsqueeze(-2) for i in range(self.n_exps)]
        else:
            expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2), expert_outputs, gates
    




class ModalFusionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, multi, img_dim, txt_dim):
        super(ModalFusionLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.multi = multi
        self.img_dim = img_dim
        self.text_dim = txt_dim

        modal1 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(in_dim, out_dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.img_dim, out_dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.text_dim, out_dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)

        self.ent_attn = nn.Linear(self.out_dim, 1, bias=False)
        self.ent_attn.requires_grad_(True)

    def forward(self, modal1_emb, modal2_emb, modal3_emb):
        batch_size = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_stack = torch.stack((x_modal1, x_modal2, x_modal3), dim=1)
            attention_scores = self.ent_attn(x_stack).squeeze(-1)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            context_vectors = torch.sum(attention_weights.unsqueeze(-1) * x_stack, dim=1)
            x_mm.append(context_vectors)
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        return x_mm, attention_weights
    

    



class MultiModelFusion(nn.Module):
    def __init__(self, in_dim, out_dim, multi, img_dim, txt_dim):
        super(MultiModelFusion, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.multi = multi
        self.img_dim = in_dim
        self.text_dim = in_dim
        self.mm_dim = in_dim
        modal1 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(in_dim, out_dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.img_dim, out_dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.text_dim, out_dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)
        
        modal4 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.mm_dim , out_dim)
            modal4.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal4_layers = nn.ModuleList(modal4)

        self.ent_attn = nn.Linear(self.out_dim, 1, bias=False)
        self.ent_attn.requires_grad_(True)

    def forward(self, modal1_emb, modal2_emb, modal3_emb, modal4_emb):
        batch_size = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_modal4 = self.modal4_layers[i](modal4_emb)
            x_stack = torch.stack((x_modal1, x_modal2, x_modal3,x_modal4), dim=1)
            attention_scores = self.ent_attn(x_stack).squeeze(-1)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            context_vectors = torch.sum(attention_weights.unsqueeze(-1) * x_stack, dim=1)
            x_mm.append(context_vectors)
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        return x_mm, attention_weights
    



class Ss_network(BaseModel):
    def __init__(self, args):
        super(Ss_network, self).__init__(args)
        self.entity_embeddings = nn.Embedding(
            len(args.entity2id),
            args.dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.entity_embeddings.weight)

        self.relation_embeddings = nn.Embedding(
            2 * len(args.relation2id), 
            args.r_dim, 
            padding_idx=None
        )
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        if args.pre_trained:
            self.entity_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float(), freeze=False)
            self.relation_embeddings = nn.Embedding.from_pretrained(torch.cat((
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float(),
                -1 * torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()), dim=0), freeze=False)

        self.rel_gate = nn.Embedding(2 * len(args.relation2id), 1, padding_idx=None)

        if args.dataset == "DB15K":
            img_pool = torch.nn.AvgPool2d(4, stride=4)
            img = img_pool(args.img.to(self.device).view(-1, 64, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)
        elif "MKG" in args.dataset:
            # multi-modal information for MKG
            img = args.img.to(self.device).view(args.img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 32))
            txt = txt.view(txt.size(0), -1)
        elif "TIVA" in args.dataset:
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 32, 64))
            img = img.view(img.size(0), -1)
            txt = args.desp.to(self.device)
            txt = txt.view(txt.size(0), -1)
        elif "Kuai" in args.dataset:
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 12, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)
        elif "WN9" in args.dataset:
            img_pool = torch.nn.AvgPool2d(4, stride=4)
            img = img_pool(args.img.to(self.device).view(-1, 64, 64))
            img = img.view(img.size(0), -1)
            img = torch.tensor(img).to(torch.float32)
            txt = args.desp.to(self.device)
            txt = txt.view(txt.size(0), -1)
            txt = torch.tensor(txt).to(torch.float32)
        elif "FB15K-237" in args.dataset:
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 12, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)

        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=True)
        self.share_img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=True)
        self.img_relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),
            args.r_dim, 
            padding_idx=None
        )
        nn.init.xavier_normal_(self.img_relation_embeddings.weight)



        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=True)
        self.share_txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=True)
        self.txt_relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),
            args.r_dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.txt_relation_embeddings.weight)

        self.dim = args.dim
        self.img_dim = self.img_entity_embeddings.weight.data.shape[1]
        self.txt_dim = self.txt_entity_embeddings.weight.data.shape[1]
        self.fuse_out_dim = self.dim
        self.share_dim = self.dim 
        self.f_dim = self.dim 
        self.TuckER_S = TuckERLayer(self.dim, args.r_dim)
        self.TuckER_I = TuckERLayer(self.dim, args.r_dim)
        self.TuckER_D = TuckERLayer(self.dim, args.r_dim)
        self.TuckER_share = TuckERLayer(self.share_dim, args.r_dim)
        self.TuckER_f = TuckERLayer(self.f_dim, args.r_dim)
        group_path = './datasets/'+args.dataset+'/group_en_'+args.gtype+'_'+str(args.group_num)+'.txt'
        with open(group_path, 'r') as file:
            loaded_data = json.load(file)
        group_t_en = torch.tensor(loaded_data)
        group_path = './datasets/'+args.dataset+'/group_txt_'+args.gtype+'_'+str(args.group_num)+'.txt'
        with open(group_path, 'r') as file:
            loaded_data = json.load(file)
        group_t_txt = torch.tensor(loaded_data)
        group_path = './datasets/'+args.dataset+'/group_img_'+args.gtype+'_'+str(args.group_num)+'.txt'
        with open(group_path, 'r') as file:
            loaded_data = json.load(file)
        group_t_img = torch.tensor(loaded_data)
        group_path = './datasets/'+args.dataset+'/group_mm_'+args.gtype+'_'+str(args.group_num)+'.txt'
        with open(group_path, 'r') as file:
            loaded_data = json.load(file)
        group_t_mm = torch.tensor(loaded_data)
        self.visual_encoder = ExpertEncoder(n_exps=args.n_exp, layers=[self.img_dim, self.dim],
                                          x_em=self.img_dim,edge_h=self.dim,gnn_h=self.img_dim,gnn_layer=2,city_num=len(args.entity2id),group_num=args.group_num,device=args.device,groupt =  group_t_img)
        self.text_encoder = ExpertEncoder(n_exps=args.n_exp, layers=[self.txt_dim, self.dim],
                                        x_em=self.txt_dim,edge_h=self.dim,gnn_h=self.txt_dim,gnn_layer=2,city_num=len(args.entity2id),group_num=args.group_num,device=args.device,groupt = group_t_txt)
        self.structure_encoder = ExpertEncoder(n_exps=args.n_exp, layers=[self.dim, self.dim],
                                             x_em=self.dim,edge_h=self.dim,gnn_h=self.dim,gnn_layer=2,city_num=len(args.entity2id),group_num=args.group_num,device=args.device,groupt =  group_t_en )
        self.share_encoder = ExpertEncoder(n_exps=args.n_exp, layers=[self.fuse_out_dim, self.fuse_out_dim],
                                         x_em=self.fuse_out_dim,edge_h=self.dim,gnn_h=self.fuse_out_dim,gnn_layer=2,city_num=len(args.entity2id),group_num=args.group_num,device=args.device,groupt = group_t_mm)



        self.spec_classifier = nn.Sequential(
            nn.Linear(args.dim, args.dim), 
            nn.ReLU(), 
            nn.Linear(args.dim, 3) 
             )
        
        self.share_classifier = nn.Sequential(
            nn.Linear(args.dim, args.dim), 
            nn.ReLU(), 
            nn.Linear(args.dim, 3) 
             )
        
        self.img_align  = nn.Linear(in_features=self.img_dim, out_features=args.dim, bias=False)
        self.txt_align  = nn.Linear(in_features=self.txt_dim, out_features=args.dim, bias=False)

        self.fuse_e_share = ModalFusionLayer(
            in_dim=args.dim,
            out_dim=self.fuse_out_dim,
            multi=2,
            img_dim=self.dim,
            txt_dim=self.dim
        )

        self.fuse_r_share = ModalFusionLayer(
            in_dim=args.r_dim,
            out_dim=self.fuse_out_dim,
            multi=2,
            img_dim=args.r_dim,
            txt_dim=args.r_dim
        )
        
    
        self.fuse_e_f = MultiModelFusion(
            in_dim=args.dim,
            out_dim=self.fuse_out_dim,
            multi=2,
            img_dim=self.dim,
            txt_dim=self.dim
        )

        self.fuse_r_f = MultiModelFusion(
            in_dim=args.r_dim,
            out_dim=self.fuse_out_dim,
            multi=2,
            img_dim=args.r_dim,
            txt_dim=args.r_dim
        )


        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()
        self.en_rgcn = RGCNCell(len(args.entity2id),self.dim,self.dim,2 * len(args.relation2id),100,100,1,0.2,True,False,"convgcn","sub",self.relation_embeddings,True,False)
        
    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        rel_gate = self.rel_gate(relation)


        e_embed, disen_str, atten_s = self.structure_encoder(self.entity_embeddings(head), rel_gate,head=head)
        r_embed = self.relation_embeddings(relation)
        e_img_embed, disen_img, atten_i = self.visual_encoder(self.img_entity_embeddings(head), rel_gate,head=head)
        r_img_embed = self.img_relation_embeddings(relation)
        e_txt_embed, disen_txt, atten_t = self.text_encoder(self.txt_entity_embeddings(head), rel_gate,head=head)
        r_txt_embed = self.txt_relation_embeddings(relation)

        """spec & share"""
        spec_s = torch.softmax(self.spec_classifier(e_embed), dim=1)
        spec_i = torch.softmax(self.spec_classifier(e_img_embed), dim=1)                                   
        spec_t = torch.softmax(self.spec_classifier(e_txt_embed), dim=1)

        share_raw_img_embed = self.img_align(self.share_img_entity_embeddings(head))
        share_raw_txt_embed = self.txt_align(self.share_txt_entity_embeddings(head))
        share_s_embed, _, _ = self.share_encoder(self.entity_embeddings(head), rel_gate,head=head)
        share_e_img_embed, _, _ = self.share_encoder(share_raw_img_embed, rel_gate,head=head)
        share_e_txt_embed, _, _ = self.share_encoder(share_raw_txt_embed, rel_gate,head=head)

        share_e_embed, _ = self.fuse_e_share(share_s_embed, share_e_img_embed, share_e_txt_embed)
        share_r_embed, _ = self.fuse_r_share(r_embed, r_img_embed, r_txt_embed)
        

        # final fusion
        f_e_embed, f_e_attn = self.fuse_e_f(e_embed, e_img_embed, e_txt_embed, share_e_embed)
        f_r_embed, f_r_attn = self.fuse_r_f(r_embed, r_img_embed, r_txt_embed, share_r_embed)

        share_s = torch.log_softmax(self.share_classifier(share_s_embed), dim=1)
        share_i = torch.log_softmax(self.share_classifier(share_e_img_embed), dim=1)
        share_t = torch.log_softmax(self.share_classifier(share_e_txt_embed), dim=1)


        """specific & share"""

        pred_s = self.TuckER_S(e_embed, r_embed)
        pred_i = self.TuckER_I(e_img_embed, r_img_embed)
        pred_d = self.TuckER_D(e_txt_embed, r_txt_embed)
        pred_share = self.TuckER_share(share_e_embed, share_r_embed)

        all_s = self.entity_embeddings.weight
        all_v = self.img_entity_embeddings.weight
        all_t = self.txt_entity_embeddings.weight

        all_s_temp = self.entity_embeddings.weight
        all_v = self.img_align(all_v)
        all_t = self.txt_align(all_t)
        all_share, _ = self.fuse_e_share(all_s_temp, all_v, all_t)


        pred_s = torch.mm(pred_s, all_s.transpose(1, 0))
        pred_i = torch.mm(pred_i, all_v.transpose(1, 0))
        pred_d = torch.mm(pred_d, all_t.transpose(1, 0))
        pred_share = torch.mm(pred_share, all_share.transpose(1, 0))

        """final fusion"""
        pred_f = self.TuckER_f(f_e_embed, f_r_embed)
        all_f, _ = self.fuse_e_f(all_s, all_v, all_t, all_share)
        pred_f = torch.mm(pred_f, all_f.transpose(1, 0))
        
  

        pred_s = F.softmax(pred_s, dim=1)
        pred_i = F.softmax(pred_i, dim=1)
        pred_d = F.softmax(pred_d, dim=1)
        pred_share = F.softmax(pred_share, dim=1)
        pred_f = F.softmax(pred_f, dim=1)


        attn_f = torch.zeros(atten_s.shape[0], dtype=torch.long).cuda()



        if not self.training:
            return [pred_s, pred_i, pred_d, pred_share, pred_f], [atten_s, atten_i, atten_t, attn_f], share_s, share_i, share_t, spec_s, spec_i, spec_t
        else:
            return [pred_s, pred_i, pred_d, pred_share, pred_f], [disen_str, disen_img, disen_txt], share_s, share_i, share_t, spec_s, spec_i, spec_t
        
    
    def get_batch_embeddings(self, batch_inputs):
        head = batch_inputs[:, 0]
        _, disen_str, _ = self.structure_encoder(self.entity_embeddings(head))
        _, disen_img, _ = self.visual_encoder(self.img_entity_embeddings(head))
        _, disen_txt, _ = self.text_encoder(self.txt_entity_embeddings(head))
        return [disen_str, disen_img, disen_txt]


    def loss_func(self, output, target):
        loss_s = self.bceloss(output[0], target)
        loss_i = self.bceloss(output[1], target)
        loss_d = self.bceloss(output[2], target)
        loss_share = self.bceloss(output[3], target)
        loss_f = self.bceloss(output[4], target)
        return loss_s + loss_i + loss_d + loss_share + loss_f
    def en_graph(self,g):
        with torch.no_grad():
            g_en_emb = F.normalize(self.en_rgcn.forward(g, self.entity_embeddings.weight, [self.relation_embeddings.weight, self.relation_embeddings.weight]))
            self.entity_embeddings.weight.data.copy_(g_en_emb)