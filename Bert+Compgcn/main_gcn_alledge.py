import scipy.sparse as sp
import math
from ordered_set import OrderedSet
from collections import namedtuple
import torch
import  torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel,BertTokenizer
import sys
import time
import argparse
import numpy as np
import os
print(torch.version.cuda)
os.environ["CUDA_VISIBLE_DEVICES"]="1"
class BERT_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/baichuanyang/bert-base-chinese/')
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


    def  __init__(self, in_features, out_features, bias=True):
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
    def __init__(self,nfeature,nhidden,nclass,dropout):
        super(GCN, self).__init__()
        self.gcn1=GCNConv(nfeature,nhidden)
        self.gcn2=GCNConv(nfeature,nhidden)
        # self.dropout = dropout
        self.fc=nn.Linear(nhidden*2,nclass)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self,index,label,sentence_mask,features,edges):
        edges=torch.stack(index,dim=0)
        # edges=edges.transpose(0,1)
        logits1_gcn=self.gcn1(features,edges)
        logits2_gcn=self.gcn2(logits1_gcn,edges)
        x = self.take_logtis(logits2_gcn,index)
        x= torch.cat(x,dim=1)
        x = self.fc(x)
        return self.criterion(x, label), x
    def take_logtis(self,logits_gcn,index):
        final_logits=[torch.index_select(logits_gcn,0,i) for i in index]
        return  final_logits

class SenGraph(nn.Module):
    def __init__(self,nfeature,nhidden,num_class,dropout):
        super(SenGraph, self).__init__()
        self.gcn=GCNConv(nfeature,nhidden)
        self.bert=BertModel.from_pretrained('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/baichuanyang/bert-base-chinese/')
        # self.bert_fc=nn.Linear(768,nhidden)
        self.class_fc=nn.Linear(nhidden*2+nhidden,num_class)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,index,label,sentence_mask,features,edges):
        # edges=torch.stack(index,dim=0)
        logits_gcn=self.gcn(features,edges)
        e1_e2=self.take_logtis(logits_gcn,index)
        sentence_encode=self.bert(sentence_mask[:,0,:],sentence_mask[:,1,:])[0][:,0,:] #cls
        # logits_sentence=self.bert_fc(sentence_encode)
        logits=torch.cat([e1_e2[0],e1_e2[1],sentence_encode],dim=1)
        output = self.class_fc(logits)
        return self.criterion(output, label), output

    def take_logtis(self,logits_gcn,index):
        final_logits=[torch.index_select(logits_gcn,0,i) for i in index]
        return  final_logits

def map_id_rel():
    rel2id = {'synonym': 0, 'hypernym': 1, 'uncorrelated': 2, 'needA': 3, 'needSpecific': 4, 'correlated': 5,
              'hasA': 6}
    id2rel = {}
    for i in rel2id:
        id2rel[rel2id[i]] = i
    return rel2id, id2rel

def load_turple_emb(args,entity_list):

    def load_emb(entity,tokenizer,max_length):
        train_data =[]
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
            train_data.append(torch.stack([indexed_tokens,att_mask]))
        return train_data
    max_length = 10  #以后改成仅仅是sub，obj来代替sentence训练 max_length降低
    tokenizer = BertTokenizer.from_pretrained('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/baichuanyang/bert-base-chinese/')
    entity_dict=load_emb(entity_list,tokenizer,max_length)

    entity_dict = torch.stack(entity_dict)

    # model=BertModel.from_pretrained('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/baichuanyang/bert-base-chinese/')
    model=BERT_Classifier()
    model.load_state_dict(torch.load("/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/baichuanyang/model/mt_cls1_simcse_unsup.pt"))
    model=model.cuda()
    model=model.eval()

    # model = torch.load('/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/Rert_re/bert-base-chinese/pytorch_model.bin')
    # state_dict = model['state_dict']
    # del state_dict['network']['fc2.weight']
    # del state_dict['network']['fc2.bias']
    # model = BERT_Classifier()
    # model.load_state_dict(state_dict['network'])
    # model = model.cuda()
    # model=model.eval()

    with torch.no_grad():
        entity_iter = torch.utils.data.DataLoader(entity_dict,64,shuffle=False)
        for i,text_mask in enumerate(entity_iter) :
            text = text_mask.squeeze(2)[:, 0, :].cuda()
            mask = text_mask.squeeze(2)[:, 1, :].cuda()
            temp=model(text,mask)
            # temp=model(text,mask).last_hidden_state[:,0]
            if i==0:
                entity_emb=torch.tensor(temp)
            else:
                entity_emb=torch.cat([entity_emb,temp],dim=0)
    return entity_emb

def load_graph_data(args):
    def load_bert_ids(sentence,tokenizer,max_length):
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
        return [indexed_tokens,att_mask]

    tokenizer = BertTokenizer.from_pretrained('/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/baichuanyang/bert-base-chinese/')
    ent_set, rel_set = OrderedSet(), OrderedSet()
    for line in open(args.graph_path):
        sub, obj, rel = map(str.lower, line.strip().split('\t'))
        ent_set.add(sub)
        rel_set.add(rel)
        ent_set.add(obj)
    args.num_class=len(rel_set)
    print("关系的类别数：",len(rel_set),"关系：",rel_set)
    ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
    rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
    edges = []
    entity_text=[ent for ent in ent2id]
    print("节点的个数",len(entity_text))

    # tuple_data={'e1_index':[0],'e2_index':[1],'label':[2],'sentence_mask':[3][0],[3][1]} #转化成列表方便Dataloader
    tuple_data=[[],[],[],[]]
    for line in open(args.graph_path):
        sub, obj, rel = map(str.lower, line.strip().split('\t'))
        sentence=""
        tuple_data[0].append(torch.tensor(ent2id[sub]))
        tuple_data[1].append(torch.tensor(ent2id[obj]))
        tuple_data[2].append(torch.tensor(rel2id[rel]))
        tuple_data[3].append(torch.stack(load_bert_ids(sentence,tokenizer,args.max_length)))
        edges.append([ent2id[sub], ent2id[obj]])
    print("边的个数",len(edges))
    edges=torch.tensor(np.array(edges,dtype="int32"))
    if args.init_mode=="yuyi":
        features=load_turple_emb(args,entity_text) #entity index 和embeding 对应（ent2id)
        print("语义初始化向量")
    if args.init_mode=="random":
        features = torch.rand(len(entity_text),200)
        print("随机初始化向量")
    return edges, tuple_data ,features,ent2id

if __name__ == '__main__':

    '''
    定义一个显示超参数的函数，将代码中所有的超参数打印
    '''
    def show_Hyperparameter(args):
        argsDict = args.__dict__
        print(argsDict)
        print('the settings are as following')
        for key in argsDict:
            print(key,':',argsDict[key])
    '''
    训练设置
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', default="cross.txt")
    parser.add_argument('--emb_path', default="/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/Rert_re/mt_data/1_768_model.pt")
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_epoch', type=int, default=400,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.000001,
                        help='Initial learning rate')
    parser.add_argument('--init_mode', type=str, default="yuyi")
    # 权重衰减
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 loss on parameters)')
    parser.add_argument('--hidden', type=int, default=768,
                        help='Number of hidden units')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_class', type=int, default=7)
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability)')

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
    edges, tuple_data,features,ent2id= load_graph_data(args)
    # Model and optimizer
    model = GCN(features[0].shape[0],
                     args.hidden,
                     args.num_class,
                     args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    # 如果可以使用GPU，数据写入cuda，便于后续加速
    # .cuda()会分配到显存里（如果gpu可用）
    if args.cuda:
        model.cuda()
        features =features.cuda()
        edges = edges.cuda()
        tuple_data[0] = torch.stack(tuple_data[0]).cuda()
        tuple_data[1] = torch.stack(tuple_data[1]).cuda()
        tuple_data[2] = torch.stack(tuple_data[2]).cuda()
        tuple_data[3] = torch.stack(tuple_data[3]).cuda()
        # adj = adj.cuda()

    def dataset_split(data,ratio):
        train_dataset = torch.utils.data.TensorDataset(data[0][:int(ratio*len(data[0]))], data[1][:int(ratio*len(data[0]))], data[2][:int(ratio*len(data[0]))], data[3][:int(ratio*len(data[0]))])
        test_dataset = torch.utils.data.TensorDataset(data[0][int(ratio*len(data[0])):], data[1][int(ratio*len(data[0])):], data[2][int(ratio*len(data[0])):], data[3][int(ratio*len(data[0])):])
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
            for e1_index,e2_index,label,sentence_mask in train_iter:
                loss, output = model([e1_index,e2_index],label,sentence_mask.squeeze(2),features,edges)
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
                        'epoch': epoch}, str(epoch) + 'bertemb_gcn_model.pt')

    def eval(args,model):
        model.eval()
        test_iter = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=True)
        with torch.no_grad():
            correct = 0
            total_num=0
            for e1_index,e2_index,label,sentence_mask in test_iter:
                loss, output = model([e1_index,e2_index],label,sentence_mask.squeeze(2),features,edges)
                _, predicted = torch.max(output.data, 1)
                correct += predicted.data.eq(label.data).cpu().sum()
                total_num+=label.shape[0]
            acc= (1.0*correct.numpy())/total_num
            print("Eval Result: right", correct.cpu().numpy().tolist(), "total", total_num, "Acc:", acc)
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
            for e1_index,e2_index,label,sentence_mask in test_iter:
                loss, output = model([e1_index,e2_index],label,sentence_mask.squeeze(2),features)
                _, predicted = torch.max(output.data, 1)
                correct += predicted.data.eq(label.data).cpu().sum()
                total_num+=label.shape[0]
                for i in range(len(label)):
                    result.append([id2ent[e1_index.cpu().numpy()[i]],id2ent[e2_index.cpu().numpy()[i]],id2rel[label.cpu().numpy()[i]],id2rel[predicted.cpu().numpy()[i]]])
            acc= (1.0*correct.numpy())/total_num
            print("Eval Result: right", correct.cpu().numpy().tolist(), "total", total_num, "Acc:", acc)
            with open("result","w",encoding='utf8') as w:
                for line in result:
                    for i in line:
                        w.write(i+'\t')
                    w.write('\n')
        return acc
    train(args)
