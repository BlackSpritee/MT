import scipy.sparse as sp
import math
from ordered_set import OrderedSet
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import sys
import time
import argparse
import numpy as np
import os

print(torch.version.cuda)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from message_passing import MessagePassing
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
from torch_scatter import scatter_add

rel2id = {'synonym': 0, 'hypernym': 1, 'uncorrelated': 2, 'needa': 3, 'needspecific': 4, 'correlated': 5,
          'hasa': 6}


def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    xavier_normal_(param.data)
    return param


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


class CompGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, act=lambda x: x, params=None):
        super(self.__class__, self).__init__()

        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act
        self.device = None

        self.w_loop = get_param((in_channels, out_channels))
        self.w_in = get_param((in_channels, out_channels))
        self.w_out = get_param((in_channels, out_channels))
        self.w_rel = get_param((in_channels, out_channels))
        self.loop_rel = get_param((1, in_channels));

        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(out_channels)

        if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

    def forward(self, x, edge_index, edge_type, rel_embed):  # x->features

        edge_index = edge_index.cuda().long()
        edge_type = edge_type.cuda().long()
        # edge_index edge_type 指的是全部边还是当前边
        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges].cuda(), edge_type[num_edges:].cuda()

        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).to(self.device)

        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)

        in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed,
                                edge_norm=self.in_norm, mode='in')
        loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed,
                                  edge_norm=None, mode='loop')
        out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed,
                                 edge_norm=self.out_norm, mode='out')
        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)

        if self.p.bias: out = out + self.bias
        out = self.bn(out)

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]  # Ignoring the self loop inserted

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight = getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)


class BERT_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            '/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/baichuanyang/bert-base-chinese/')
        # self.fc1 = nn.Linear(768, 200)

    def forward(self, sentence_ids, mask):
        x = self.bert(sentence_ids, attention_mask=mask)[0]
        x = x[:, 0, :]
        # x = self.fc1(x)
        return x


class GraphConvolution(nn.Module):  # Module类的单继承
    """
    简单的GCN层，类似于https://arxiv.org/abs/1609.02907
    """

    """
    参数：
        in_features：输入特征，每个输入样本的大小
        out_features：输出特征，每个输出样本的大小
        bias：偏置，如果设置为False，则层将不会学习加法偏差。默认值：True
    属性：
        weight：形状模块的可学习权重（out_features x in_features）
        bias：形状模块的可学习偏差（out_features）	
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        # super()函数是用于调用父类(超类)的方法
        # super().__init__()表示子类既能重写__init__()方法又能调用父类的方法

        self.in_features = in_features
        self.out_features = out_features

        ##################参数定义##############################
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # 先转化为张量，再转化为可训练的Parameter对象
        # Parameter用于将参数自动加入到参数列表
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)  # 第一个参数必须按照字符串形式输入
            # 将Parameter对象通过register_parameter()进行注册
            # 为模型添加参数
            self.reset_parameters()

    def reset_parameters(self):  # 参数随机初始化函数
        stdv = 1. / math.sqrt(self.weight.size(1))
        # size包括(in_features, out_features)，size(1)应该是指out_features
        # stdv=1/根号(out_features)
        self.weight.data.uniform_(-stdv, stdv)
        # weight在区间(-stdv, stdv)之间均匀分布随机初始化
        if self.bias is not None:  # 变量是否不是None
            self.bias.data.uniform_(-stdv, stdv)  # bias均匀分布随机初始化

    def forward(self, input, adj):  # 前向传播函数
        support = torch.mm(input, self.weight)
        # input和self.weight矩阵相乘
        output = torch.spmm(adj, support)
        # spmm()是稀疏矩阵乘法，说白了还是乘法而已，只是减小了运算复杂度
        # 最新spmm函数移到了torch.sparse模块下，但是不能用
        if self.bias is not None:  # 变量是否不是None
            return output + self.bias  # 返回（系数*输入*权重+偏置）
        else:
            return output  # 返回（系数*输入*权重）无偏置

    def __repr__(self):  # 打印输出
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    # 打印形式是：GraphConvolution (输入特征 -> 输出特征)


# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, dropout):
#         super(GCN, self).__init__()
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         # x = F.relu(self.gc1(x, adj))
#         x=self.gc1(x,adj)
#         x = F.dropout(x, self.dropout)
#         return x

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GATConv


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out


class GCN(nn.Module):
    def __init__(self, nfeature, nhidden, nclass, n_edge_type, args):
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(nfeature, nhidden)
        self.init_rel = get_param((n_edge_type, nfeature))  # 边的类别：节点类别-节点类别
        self.compgcn = CompGCNConv(nfeature, nhidden, nclass, torch.tanh, params=args)
        # self.dropout = dropout
        self.fc = nn.Linear(nhidden * 2, nclass)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, index, label, sentence_mask, edges_type, features, edges):
        # edges=torch.stack(index,dim=0)
        edges = edges.transpose(0, 1)
        r = self.init_rel
        # logits_gcn=self.gcn(features,edges)
        x, r = self.compgcn(features, edges, edges_type, r)
        x = self.take_logtis(x, index)
        x = torch.cat(x, dim=1)
        x = self.fc(x)
        return self.criterion(x, label), x

    def take_logtis(self, logits_gcn, index):
        final_logits = [torch.index_select(logits_gcn, 0, i) for i in index]
        return final_logits


class SenGraph(nn.Module):
    def __init__(self, nfeature, nhidden, num_class, dropout):
        super(SenGraph, self).__init__()
        self.gcn = GCNConv(nfeature, nhidden)
        self.bert = BertModel.from_pretrained(
            '/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/baichuanyang/bert-base-chinese/')
        # self.bert_fc=nn.Linear(768,nhidden)
        self.class_fc = nn.Linear(nhidden * 2 + nhidden, num_class)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, index, label, sentence_mask, features):
        edges = torch.stack(index, dim=0)
        logits_gcn = self.gcn(features, edges)
        e1_e2 = self.take_logtis(logits_gcn, index)
        sentence_encode = self.bert(sentence_mask[:, 0, :], sentence_mask[:, 1, :])[0][:, 0, :]  # cls
        # logits_sentence=self.bert_fc(sentence_encode)
        logits = torch.cat([e1_e2[0], e1_e2[1], sentence_encode], dim=1)
        output = self.class_fc(logits)
        return self.criterion(output, label), output

    def take_logtis(self, logits_gcn, index):
        final_logits = [torch.index_select(logits_gcn, 0, i) for i in index]
        return final_logits


def map_id_rel():
    id2rel = {}
    for i in rel2id:
        id2rel[rel2id[i]] = i
    return rel2id, id2rel


def load_turple_emb(args, entity_list):
    def load_emb(entity, tokenizer, max_length):
        train_data = []
        for e in entity:
            sent = str(e)
            indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
            avai_len = len(indexed_tokens)
            while len(indexed_tokens) < max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[: max_length]
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)
            # Attention mask
            att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            train_data.append(torch.stack([indexed_tokens, att_mask]))
        return train_data

    max_length = 10  # 以后改成仅仅是sub，obj来代替sentence训练 max_length降低
    tokenizer = BertTokenizer.from_pretrained(
        '/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/baichuanyang/bert-base-chinese/')
    entity_dict = load_emb(entity_list, tokenizer, max_length)

    entity_dict = torch.stack(entity_dict)

    model = BERT_Classifier()
    model.load_state_dict(
        # torch.load("/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/baichuanyang/model/mt_cls1_simcse_unsup.pt"))
        torch.load("/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/baichuanyang/model/mt_cls1_simcse_unsup.pt"))
    # model=BertModel.from_pretrained('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/baichuanyang/bert-base-chinese/')
    model = model.cuda()
    model = model.eval()

    # model = torch.load('/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/Rert_re/bert-base-chinese/pytorch_model.bin')
    # state_dict = model['state_dict']
    # del state_dict['network']['fc2.weight']
    # del state_dict['network']['fc2.bias']
    # model = BERT_Classifier()
    # model.load_state_dict(state_dict['network'])
    # model = model.cuda()
    # model=model.eval()

    with torch.no_grad():
        entity_iter = torch.utils.data.DataLoader(entity_dict, 64, shuffle=False)
        for i, text_mask in enumerate(entity_iter):
            text = text_mask.squeeze(2)[:, 0, :].cuda()
            mask = text_mask.squeeze(2)[:, 1, :].cuda()
            temp = model(text, mask)
            # temp=model(text,mask).last_hidden_state[:,0]
            if i == 0:
                entity_emb = torch.tensor(temp)
            else:
                entity_emb = torch.cat([entity_emb, temp], dim=0)
    return entity_emb


def load_graph_data(args):
    def load_bert_ids(sentence, tokenizer, max_length):
        sent = str(sentence)
        indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
        avai_len = len(indexed_tokens)
        while len(indexed_tokens) < max_length:
            indexed_tokens.append(0)  # 0 is id for [PAD]
        indexed_tokens = indexed_tokens[: max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)
        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1
        return [indexed_tokens, att_mask]

    tokenizer = BertTokenizer.from_pretrained(
        '/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/baichuanyang/bert-base-chinese/')
    ent_set, rel_set, edge_type_set = OrderedSet(), OrderedSet(), OrderedSet()
    for line in open(args.graph_path):
        sub, sub_type, obj, obj_type, rel, _, _, _ = map(str.lower, line.strip().split('\t'))
        ent_set.add(sub)
        rel_set.add(rel)
        ent_set.add(obj)
        edge_type_set.add(sub_type + obj_type)
        edge_type_set.add(obj_type+sub_type)
    for line in open(args.predict_data):
        sub, sub_type, obj, obj_type = map(str.lower, line.strip().split('\t'))
        edge_type_set.add(sub_type+obj_type)
        edge_type_set.add(obj_type+sub_type)
    args.num_class = len(rel_set)
    print("关系的类别数：", len(rel_set), "关系：", rel_set)
    ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
    # rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
    reltype2id = {reltype: idx for idx, reltype in enumerate(edge_type_set)}
    print(reltype2id)
    edges = []
    edges_type = []
    entity_text = [ent for ent in ent2id]
    print("节点的个数", len(entity_text))

    # tuple_data={'e1_index':[0],'e2_index':[1],'label':[2],'sentence_mask':[3][0],[3][1]} #转化成列表方便Dataloader
    tuple_data = [[], [], [], [], []]
    for line in open(args.graph_path):
        sub, sub_type, obj, obj_type, rel, _, _, _ = map(str.lower, line.strip().split('\t'))
        sentence = ""
        tuple_data[0].append(torch.tensor(ent2id[sub]))
        tuple_data[1].append(torch.tensor(ent2id[obj]))
        tuple_data[2].append(torch.tensor(rel2id[rel]))
        tuple_data[3].append(torch.stack(load_bert_ids(sentence, tokenizer, args.max_length)))
        tuple_data[4].append(torch.tensor(reltype2id[sub_type + obj_type]))
        edges.append([ent2id[sub], ent2id[obj]])
        edges_type.append(reltype2id[sub_type + obj_type])
    for line in open(args.predict_data):
        sub, sub_type, obj, obj_type = map(str.lower, line.strip().split('\t'))
        edges.append([ent2id[sub], ent2id[obj]])
        edges_type.append(reltype2id[sub_type + obj_type])
    print("边的个数", len(edges))
    if args.init_mode == "yuyi":
        features = load_turple_emb(args, entity_text)  # entity index 和embeding 对应（ent2id)
        print("语义初始化向量")
    if args.init_mode == "random":
        features = torch.rand(len(entity_text), 200)
        print("随机初始化向量")

    if args.predict_mode == True:
        predict_data = [[], [], [], [], []]
        # predict_edges = []
        # predict_edges_type = []
        predict_edges = edges
        predict_edges_type = edges_type
        for line in open(args.predict_data):
            sub, sub_type, obj, obj_type = map(str.lower, line.strip().split('\t'))
            sentence = ""
            predict_data[0].append(torch.tensor(ent2id[sub]))
            predict_data[1].append(torch.tensor(ent2id[obj]))
            predict_data[2].append(torch.tensor(rel2id["uncorrelated"]))
            predict_data[3].append(torch.stack(load_bert_ids(sentence, tokenizer, args.max_length)))
            predict_data[4].append(torch.tensor(reltype2id[sub_type + obj_type]))
            # predict_edges.append([ent2id[sub], ent2id[obj]])
            # predict_edges_type.append(reltype2id[sub_type + obj_type])
        print("边的个数", len(predict_edges))
        print(edges[240600:241000])
        predict_edges = torch.tensor(np.array(predict_edges, dtype="int32"))
        predict_edges_type = torch.tensor(np.array(predict_edges_type, dtype="int32"))
        return edges, tuple_data, features, ent2id, edges_type, len(
            reltype2id), predict_data, predict_edges, predict_edges_type
    else:
        return edges, tuple_data, features, ent2id, edges_type, len(reltype2id), None, None, None


if __name__ == '__main__':

    '''
    定义一个显示超参数的函数，将代码中所有的超参数打印
    '''


    def show_Hyperparameter(args):
        argsDict = args.__dict__
        print(argsDict)
        print('the settings are as following')
        for key in argsDict:
            print(key, ':', argsDict[key])


    '''
    训练设置
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', default="cross_24w.txt")
    parser.add_argument('--emb_path',
                        default="/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/Rert_re/mt_data/1_768_model.pt")
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Disables CUDA training.')

    parser.add_argument('--predict_mode', action='store_true', default=True)
    parser.add_argument('--predict_data', action='store_true', default="predict_ingraph.txt")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_epoch', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='Initial learning rate')
    parser.add_argument('--init_mode', type=str, default="yuyi")
    # 权重衰减
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 loss on parameters)')
    parser.add_argument('--hidden', type=int, default=768,
                        help='Number of hidden units')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_class', type=int, default=7)
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--opn', type=str, default='corr')
    parser.add_argument('--bias', type=bool, default=True)
    args = parser.parse_args()
    show_Hyperparameter(args)
    args.cuda = args.cuda and torch.cuda.is_available()
    # 指定生成随机数的种子，从而每次生成的随机数都是相同的，通过设定随机数种子的好处是，使模型初始化的可学习参数相同，从而使每次的运行结果可以复现。
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    '''
    开始训练
    '''

    # 载入数据
    edges, tuple_data, features, ent2id, edges_type, len_reltype, predict_turple_data, predict_edges, predict_edges_type = load_graph_data(
        args)
    # Model and optimizer
    model = GCN(features[0].shape[0],
                args.hidden,
                args.num_class,
                len_reltype,
                args)
    if args.predict_mode == True:
        if args.cuda:
            features = features.cuda()
            predict_edges_type = predict_edges_type.cuda()
            predict_turple_data[0] = torch.stack(predict_turple_data[0]).cuda()
            predict_turple_data[1] = torch.stack(predict_turple_data[1]).cuda()
            predict_turple_data[2] = torch.stack(predict_turple_data[2]).cuda()
            predict_turple_data[3] = torch.stack(predict_turple_data[3]).cuda()
            predict_turple_data[4] = torch.stack(predict_turple_data[4]).cuda()
        predict_dataset = torch.utils.data.TensorDataset(predict_turple_data[0], predict_turple_data[1],
                                                         predict_turple_data[2], predict_turple_data[3],
                                                         predict_turple_data[4])
        model.load_state_dict(torch.load(
            # "/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/baichuanyang/project/verniesage3/verniesage2/cora/29bertemb_compgcn_model.pt")['state_dict']['network'])
            "/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/baichuanyang/project/verniesage3/verniesage2/cora/11_24w_ugcbert_corr_compgcn2.pt")['state_dict']['network'])
        _, id2rel = map_id_rel()
        id2ent = {}
        result = []
        for i in ent2id:
            id2ent[ent2id[i]] = i
        model = model.cuda()
        model = model.eval()
        predict_iter = torch.utils.data.DataLoader(predict_dataset, args.batch_size, shuffle=True)
        with torch.no_grad():
            for e1_index, e2_index, label, sentence_mask, _ in predict_iter:
                loss, output = model([e1_index, e2_index], label, sentence_mask.squeeze(2), predict_edges_type,
                                     features, predict_edges)
                output=nn.functional.softmax(output.data, 1)
                score, predicted = torch.max(output, 1)
                for i in range(len(label)):
                    result.append([id2ent[e1_index.cpu().numpy()[i]], id2ent[e2_index.cpu().numpy()[i]],
                                   id2rel[label.cpu().numpy()[i]], id2rel[predicted.cpu().numpy()[i]],
                                   score.cpu().numpy()[i]])
        with open("result_predict_24w_ugcbert", "w", encoding='utf8') as w:
            for line in result:
                print(line)
                for i in line:
                    w.write(str(i) + '\t')
                w.write('\n')

'''   optimizer = optim.Adam(model.parameters(),
                          lr=args.lr, weight_decay=args.weight_decay)

   # 如果可以使用GPU，数据写入cuda，便于后续加速
   # .cuda()会分配到显存里（如果gpu可用）
   if args.cuda:
       model.cuda()
       features =features.cuda()
       edges_type=edges_type.cuda()
       tuple_data[0] = torch.stack(tuple_data[0]).cuda()
       tuple_data[1] = torch.stack(tuple_data[1]).cuda()
       tuple_data[2] = torch.stack(tuple_data[2]).cuda()
       tuple_data[3] = torch.stack(tuple_data[3]).cuda()
       tuple_data[4] = torch.stack(tuple_data[4]).cuda()
       # adj = adj.cuda()

   def dataset_split(data,ratio):
       train_dataset = torch.utils.data.TensorDataset(data[0][:int(ratio*len(data[0]))], data[1][:int(ratio*len(data[0]))], data[2][:int(ratio*len(data[0]))], data[3][:int(ratio*len(data[0]))],data[4][:int(ratio*len(data[0]))])
       test_dataset = torch.utils.data.TensorDataset(data[0][int(ratio*len(data[0])):], data[1][int(ratio*len(data[0])):], data[2][int(ratio*len(data[0])):], data[3][int(ratio*len(data[0])):],data[4][int(ratio*len(data[0])):])
       # train_dataset = torch.utils.data.TensorDataset(data[0][:39000], data[1][:39000], data[2][:39000], data[3][:39000])
       # test_dataset = torch.utils.data.TensorDataset(data[0][39000:], data[1][39000:], data[2][39000:], data[3][39000:])
       return train_dataset,test_dataset
   
   train_data,test_data=dataset_split(tuple_data,ratio=0.8)

   def train(args):
       model.train()
       train_iter = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=False)
       for epoch in range (args.num_epoch):
           correct = 0
           total_num = 0
           total_loss=0
           optimizer.zero_grad()
           for e1_index,e2_index,label,sentence_mask,_ in train_iter:
               loss, output = model([e1_index,e2_index],label,sentence_mask.squeeze(2),edges_type,features,edges)
               loss.backward()
               optimizer.step()
               _, predicted = torch.max(output.data, 1)
               correct += predicted.data.eq(label.data).cpu().sum()
               total_num+=label.shape[0]
               total_loss+=loss.detach().cpu().numpy().tolist()
           loss = loss.detach().cpu()
           # torch.save(model,str(epoch)+'.pt')
           print("epoch ", str(epoch),"loss:",total_loss/total_num*args.batch_size,"    acc:", correct.cpu().numpy().tolist()/total_num)
           # torch.save({'state_dict': {'network': model.state_dict(), 'optimizer': optimizer.state_dict()},
           #                 'epoch': epoch}, str(epoch) + '_200_model.pt')
           eval(args,model)
       predict(args,model)
       torch.save({'state_dict': {'network': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       'epoch': epoch}, str(epoch) + 'bertemb_compgcn_model.pt')

   def eval(args,model):
       model.eval()
       _,id2rel=map_id_rel()
       id2ent = {}
       total_loss=0
       for i in ent2id:
           id2ent[ent2id[i]] = i
       test_iter = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=True)
       result=[]
       with torch.no_grad():
           correct = 0
           total_num=0
           for e1_index,e2_index,label,sentence_mask,_ in test_iter:
               loss, output = model([e1_index,e2_index],label,sentence_mask.squeeze(2),edges_type,features,edges)
               _, predicted = torch.max(output.data, 1)
               correct += predicted.data.eq(label.data).cpu().sum()
               total_num+=label.shape[0]
               total_loss+=loss.detach().cpu().numpy().tolist()
               for i in range(len(label)):
                   result.append([id2ent[e1_index.cpu().numpy()[i]],id2ent[e2_index.cpu().numpy()[i]],id2rel[label.cpu().numpy()[i]],id2rel[predicted.cpu().numpy()[i]]])
       acc= (1.0*correct.numpy())/total_num
       print("Eval Result: right", total_loss/total_num*args.batch_size, "total", total_num, "Acc:", acc)
       with open("result_eval_simcse","w",encoding='utf8') as w:
           for line in result:
               for i in line:
                   w.write(i+'\t')
               w.write('\n')
       return acc

   def predict(args,model):
       _,id2rel=map_id_rel()
       id2ent = {}
       result=[]
       for i in ent2id:
           id2ent[ent2id[i]] = i
       model = model.eval()
       test_iter = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=True)
       with torch.no_grad():
           correct = 0
           total_num=0
           for e1_index,e2_index,label,sentence_mask,_ in test_iter:
               loss, output = model([e1_index,e2_index],label,sentence_mask.squeeze(2),edges_type,features,edges)
               _, predicted = torch.max(output.data, 1)
               correct += predicted.data.eq(label.data).cpu().sum()
               total_num+=label.shape[0]
               for i in range(len(label)):
                   result.append([id2ent[e1_index.cpu().numpy()[i]],id2ent[e2_index.cpu().numpy()[i]],id2rel[label.cpu().numpy()[i]],id2rel[predicted.cpu().numpy()[i]]])
           acc= (1.0*correct.numpy())/total_num
           print("Eval Result: right", correct.cpu().numpy().tolist(), "total", total_num, "Acc:", acc)
           with open("result_end","w",encoding='utf8') as w:
               for line in result:
                   for i in line:
                       w.write(i+'\t')
                   w.write('\n')
       return acc

   train(args)
'''
