import argparse
import time

import dgl
import numpy as np
import torch
from tqdm import tqdm

from models.Ss_network import Ss_network
from models.model import *
from utils.data_loader import *
from utils.data_util import load_data
from utils.model_util import comp_deg_norm

def parse_args():
    config_args = {
        'lr': 0.0005,
        'dropout_gat': 0.3,
        'dropout': 0.3,
        'cuda': 0,
        'epochs_gat': 3000,
        'epochs': 2000,
        'weight_decay_gat': 1e-5,
        'weight_decay': 0,
        'seed': 43,
        'model': 'Ss_network',
        'num-layers': 3,
        'dim': 200,
        'r_dim': 200,
        'k_w': 10,
        'k_h': 20,
        'n_heads': 2,
        'dataset': 'DB15K',
        'pre_trained': 0,
        'encoder': 0,
        'image_features': 1,
        'text_features': 1,
        'patience': 5,
        'eval_freq': 100,
        'lr_reduce_freq': 500,
        'gamma': 1.0,
        'bias': 1,
        'neg_num': 2,
        'neg_num_gat': 2,
        'alpha': 0.2,
        'alpha_gat': 0.2,
        'out_channels': 32,
        'kernel_size': 3,
        'batch_size': 1024,
        'save': 1,
        'n_exp': 1,
        'img_dim': 256,
        'txt_dim': 256,
        'lambda2':0.05,
        'lambda1':0.02,
        'group_num':4,
        'gtype':'gmm',
        'rgcngraph':True
    }

    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        parser.add_argument(f"--{param}", default=val, type=type(val))
    args = parser.parse_args()
    return args

args = parse_args()
print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
print(f'Using: {args.device}')
torch.cuda.set_device(args.cuda)
for k, v in list(vars(args).items()):
    print(str(k) + ':' + str(v))

entity2id, relation2id, img_features, text_features, train_data, val_data, test_data = load_data(args.dataset)
print("Training data {:04d}".format(len(train_data[0])))

corpus = ConvECorpus(args, train_data, val_data, test_data, entity2id, relation2id)

if args.image_features:
    args.img = F.normalize(torch.Tensor(img_features), p=2, dim=1)
if args.text_features:
    args.desp = F.normalize(torch.Tensor(text_features), p=2, dim=1)
args.entity2id = entity2id
args.relation2id = relation2id

model_name = {
    'Ss_network': Ss_network
}
time.sleep(5)


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)  
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # 偏置初始化为零
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def train_decoder(args):
    src = [triplet[0] for triplet in train_data[0]]
    dst = [triplet[2] for triplet in train_data[0]]
    src_2 = src + dst
    dst_2 = dst + src
    r_num =len(args.relation2id)
    rell = [triplet[1] for triplet in train_data[0]]
    rell_1 = [i + r_num for i in rell]
    rell_2 = rell + rell_1
    rel = np.array(rell_2)
    num_nodes = len(args.entity2id)
    input_ids = torch.arange(len(args.entity2id)).to(args.device)
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src_2, dst_2)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)
    g=g.to(args.device)
    model = model_name[args.model](args)


    init_weights(model)
    args.img_dim = model.img_dim
    args.txt_dim = model.txt_dim
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    ce_criterion = nn.CrossEntropyLoss()

    print(f'Total number of parameters: {tot_params}')
    if args.cuda is not None and int(args.cuda) >= 0:
        model = model.to(args.device)
    if args.rgcngraph:
        model.en_graph(g)
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = model.init_metric_dict()
    corpus.batch_size = args.batch_size
    corpus.neg_num = args.neg_num
    training_range = tqdm(range(args.epochs))
    for epoch in training_range:
        model.train()
        epoch_loss = []
        epoch_mi_loss = []
        t = time.time()
        corpus.shuffle()
        
        for batch_num in range(corpus.max_batch_num):
            optimizer.zero_grad()
            train_indices, train_values = corpus.get_batch(batch_num)
            train_indices = torch.LongTensor(train_indices)
            if args.cuda is not None and int(args.cuda) >= 0:
                train_indices = train_indices.to(args.device)
                train_values = train_values.to(args.device)
            output, embeddings, share_s, share_i, share_t, spec_s, spec_i, spec_t = model.forward(train_indices)

            spec_label_s = torch.Tensor([[1, 0, 0]]*len(train_indices)).to(args.device)
            spec_label_i = torch.Tensor([[0, 1, 0]]*len(train_indices)).to(args.device)
            spec_label_t = torch.Tensor([[0, 0, 1]]*len(train_indices)).to(args.device)
            share_label_s = torch.Tensor([[1/3, 1/3, 1/3]]*len(train_indices)).to(args.device)
            share_label_i = torch.Tensor([[1/3, 1/3, 1/3]]*len(train_indices)).to(args.device)
            share_label_t = torch.Tensor([[1/3, 1/3, 1/3]]*len(train_indices)).to(args.device)
            
            spec_loss = ce_criterion(spec_s, spec_label_s) + ce_criterion(spec_i, spec_label_i) + ce_criterion(spec_t, spec_label_t)
            share_loss = F.kl_div(share_s, share_label_s) + F.kl_div(share_i, share_label_i) + F.kl_div(share_t, share_label_t)
            loss = model.loss_func(output, train_values) + args.lambda2 * share_loss +args.lambda1 * spec_loss
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.data.item())
            epoch_mi_loss.append(0.0)
        training_range.set_postfix(loss="main: {:.5} mi: {:.5}".format(sum(epoch_loss), sum(epoch_mi_loss)))
        lr_scheduler.step()

        if (epoch + 1) % args.eval_freq == 0:
            print("Epoch {:04d} , average loss {:.4f} , epoch_time {:.4f}\n".format(
                epoch + 1, sum(epoch_loss) / len(epoch_loss), time.time() - t))
            model.eval()
            with torch.no_grad():
                val_metrics, _ = corpus.get_validation_pred(model, 'test')
                val_metrics_s, _ = corpus.get_validation_pred_signle(model, 'test', 0)
                val_metrics_i, _ = corpus.get_validation_pred_signle(model, 'test', 1)
                val_metrics_t, _ = corpus.get_validation_pred_signle(model, 'test', 2)
                val_metrics_share, _ = corpus.get_validation_pred_signle(model, 'test', 3)
                val_metrics_mm, _ = corpus.get_validation_pred_signle(model, 'test', 4)
            if val_metrics['MRR'] > best_test_metrics['MRR']:
                best_test_metrics['MRR'] = val_metrics['MRR']
            if val_metrics['MR'] < best_test_metrics['MR']:
                best_test_metrics['MR'] = val_metrics['MR']
            if val_metrics['Hits@1'] > best_test_metrics['Hits@1']:
                best_test_metrics['Hits@1'] = val_metrics['Hits@1']
            if val_metrics['Hits@3'] > best_test_metrics['Hits@3']:
                best_test_metrics['Hits@3'] = val_metrics['Hits@3']
            if val_metrics['Hits@10'] > best_test_metrics['Hits@10']:
                best_test_metrics['Hits@10'] = val_metrics['Hits@10']
            if val_metrics['Hits@100'] > best_test_metrics['Hits@100']:
                best_test_metrics['Hits@100'] = val_metrics['Hits@100']
            print('\n'.join(['Epoch: {:04d}'.format(epoch + 1), model.format_metrics(val_metrics, 'test')]))
            print('\n'.join(['Epoch: {:04d}, Structure: '.format(epoch + 1), model.format_metrics(val_metrics_s, 'test')]))
            print('\n'.join(['Epoch: {:04d}, Image: '.format(epoch + 1), model.format_metrics(val_metrics_i, 'test')]))
            print('\n'.join(['Epoch: {:04d}, Text: '.format(epoch + 1), model.format_metrics(val_metrics_t, 'test')]))
            print('\n'.join(['Epoch: {:04d}, Share: '.format(epoch + 1), model.format_metrics(val_metrics_share, 'test')]))
            print('\n'.join(['Epoch: {:04d}, Multi-modal: '.format(epoch + 1), model.format_metrics(val_metrics_mm, 'test')]))
            print("\n\n")


    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        with torch.no_grad():
            best_test_metrics, _ = corpus.get_validation_pred(model, 'test')
    print('\n'.join(['Val set results:', model.format_metrics(best_val_metrics, 'val')]))
    print('\n'.join(['Test set results:', model.format_metrics(best_test_metrics, 'test')]))
    print("\n\n\n\n\n\n")

    if args.save:
        torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/{args.model}_{epoch}.pth')
        print('Saved model!')


if __name__ == '__main__':
    train_decoder(args)
