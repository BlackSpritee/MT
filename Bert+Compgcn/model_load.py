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
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

print(torch.version.cuda)
os.environ["CUDA_VISIBLE_DEVICES"]="2"
class BERT_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BertModel.from_pretrained('/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/Rert_re/bert-base-chinese/')
        # self.fc1 = nn.Linear(768, 200)
    def forward(self, sentence_ids, mask):
        x = self.encoder(sentence_ids, attention_mask=mask)[0]
        x = x[:, 0, :]
        # x = self.fc1(x)
        return x

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
    def __init__(self,nfeature,nhidden):
        super(GCN, self).__init__()
        self.gcn=GCNConv(nfeature,nhidden)
        # self.criterion = nn.CrossEntropyLoss()
    def forward(self,features,edges):
        # edges=torch.stack(index,dim=0)
        logits_gcn=self.gcn(features,edges)
        return logits_gcn

class SenGraph(nn.Module):
    def __init__(self,nfeature,nhidden,num_class):
        super(SenGraph, self).__init__()
        self.bert=BertModel.from_pretrained('/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/Rert_re/bert-base-chinese/')
        # self.bert_fc=nn.Linear(nfeature,nhidden)
        self.mlp1=nn.Sequential(nn.Linear(nfeature*2,768),
                                nn.ReLU())
        self.mlp2=nn.Sequential(nn.Linear(nfeature,1024),
                               nn.ReLU(),
                               nn.Linear(1024,nfeature),
                               nn.ReLU())
        self.class_fc=nn.Linear(nhidden*2,num_class)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,index,label,sentence_mask,features):
        e1_e2=self.take_logtis(features,index)
        sentence_encode=self.bert(sentence_mask[:,0,:],sentence_mask[:,1,:])[0][:,0,:] #cls
        # logits_sentence=self.bert_fc(sentence_encode)
        e1_e2=torch.cat([e1_e2[0],e1_e2[1]],dim=1)
        e1_e2=self.mlp1(e1_e2) #?????????
        e1_e2=self.mlp2(e1_e2)
        sentence_encode=self.mlp2(sentence_encode)
        logits=torch.cat([e1_e2,sentence_encode],dim=1)
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
    max_length = 10  #?????????????????????sub???obj?????????sentence?????? max_length??????
    tokenizer = BertTokenizer.from_pretrained('/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/Rert_re/bert-base-chinese/')
    entity_dict=load_emb(entity_list,tokenizer,max_length)

    entity_dict = torch.stack(entity_dict)

    model = torch.load(args.emb_path)
    state_dict = model['state_dict']
    del state_dict['network']['fc2.weight']
    del state_dict['network']['fc2.bias']
    model = BERT_Classifier()
    model.load_state_dict(state_dict['network'])
    model = model.cuda()
    model=model.eval()
    # dataset = torch.utils.data.TensorDataset(entity_dict)
    with torch.no_grad():
        entity_iter = torch.utils.data.DataLoader(entity_dict,64,shuffle=False)
        for i,text_mask in enumerate(entity_iter) :
            text = text_mask.squeeze(2)[:, 0, :].cuda()
            mask = text_mask.squeeze(2)[:, 1, :].cuda()
            temp=model(text,mask)
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

    tokenizer = BertTokenizer.from_pretrained('/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/Rert_re/bert-base-chinese/')
    ent_set, rel_set = OrderedSet(), OrderedSet()
    for line in open(args.graph_path):
        sub, obj, rel,_ = map(str.lower, line.strip().split('\t'))
        ent_set.add(sub)
        rel_set.add(rel)
        ent_set.add(obj)
    args.num_class=len(rel_set)
    print("?????????????????????",len(rel_set),"?????????",rel_set)
    ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
    rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
    edges = []
    entity_text=[ent for ent in ent2id]
    print("???????????????",len(entity_text))

    # tuple_data={'e1_index':[0],'e2_index':[1],'label':[2],'sentence_mask':[3][0],[3][1]} #?????????????????????Dataloader
    tuple_data=[[],[],[],[]]
    for line in open(args.graph_path):
        sub, obj, rel,sentence = map(str.lower, line.strip().split('\t'))
        tuple_data[0].append(torch.tensor(ent2id[sub]))
        tuple_data[1].append(torch.tensor(ent2id[obj]))
        tuple_data[2].append(torch.tensor(rel2id[rel]))
        tuple_data[3].append(torch.stack(load_bert_ids(sentence,tokenizer,args.max_length)))
        edges.append([ent2id[sub], ent2id[obj]])
    print("????????????",len(edges))
    edges=np.array(edges,dtype="int32")
    edges=np.transpose(edges)
    if args.init_mode=="yuyi":
        features=load_turple_emb(args,entity_text) #entity index ???embeding ?????????ent2id)
        print("?????????????????????")
    if args.init_mode=="random":
        features = torch.rand(len(entity_text),200)
        print("?????????????????????")

    #???gcn?????? ??????features
    # entity_emb = torch.stack(features)
    print("??????features.....")
    model = torch.load("/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/verniesage3/cora/199bertemb_gcn_model.pt")
    state_dict = model['state_dict']
    del state_dict['network']['fc.weight']
    del state_dict['network']['fc.bias']
    model = GCN(features[0].shape[0],args.hidden)
    model.load_state_dict(state_dict['network'])
    model = model.cuda()
    model = model.eval()
    with torch.no_grad():
        features=model(features,torch.tensor(edges).cuda())

    return edges, tuple_data ,features,ent2id

if __name__ == '__main__':

    '''
    ???????????????????????????????????????????????????????????????????????????
    '''
    def show_Hyperparameter(args):
        argsDict = args.__dict__
        print(argsDict)
        print('the settings are as following')
        for key in argsDict:
            print(key,':',argsDict[key])
    '''
    ????????????
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', default="/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/Rert_re/mt_data/ugc.txt")
    parser.add_argument('--emb_path', default="/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/Rert_re/mt_data/5_768_model.pt")
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_epoch', type=int, default=200,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--init_mode', type=str, default="yuyi")
    # ????????????
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 loss on parameters)')
    parser.add_argument('--hidden', type=int, default=768,
                        help='Number of hidden units')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_class', type=int, default=7)
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability)')

    args = parser.parse_args()
    show_Hyperparameter(args)
    args.cuda = args.cuda and torch.cuda.is_available()
    # ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    '''
    ????????????
    '''

    # ????????????
    edges, tuple_data,features,ent2id= load_graph_data(args)
    # Model and optimizer
    model = SenGraph(features[0].shape[0],
                args.hidden,
                args.num_class)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    # ??????????????????GPU???????????????cuda?????????????????????
    # .cuda()??????????????????????????????gpu?????????
    if args.cuda:
        model.cuda()
        features =features.cuda()
        tuple_data[0] = torch.stack(tuple_data[0]).cuda()
        tuple_data[1] = torch.stack(tuple_data[1]).cuda()
        tuple_data[2] = torch.stack(tuple_data[2]).cuda()
        tuple_data[3] = torch.stack(tuple_data[3]).cuda()
        # adj = adj.cuda()

    def dataset_split(data,ratio):
        # train_dataset = torch.utils.data.TensorDataset(data[0][:int(ratio*len(data[0]))], data[1][:int(ratio*len(data[0]))], data[2][:int(ratio*len(data[0]))], data[3][:int(ratio*len(data[0]))])
        # test_dataset = torch.utils.data.TensorDataset(data[0][int(ratio*len(data[0])):], data[1][int(ratio*len(data[0])):], data[2][int(ratio*len(data[0])):], data[3][int(ratio*len(data[0])):])
        train_dataset = torch.utils.data.TensorDataset(data[0][:39000], data[1][:39000], data[2][:39000], data[3][:39000])
        test_dataset = torch.utils.data.TensorDataset(data[0][39000:], data[1][39000:], data[2][39000:], data[3][39000:])
        return train_dataset,test_dataset

    train_data,test_data=dataset_split(tuple_data,ratio=0.8)

    def train(args):
        model.train()
        train_iter = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=False)
        for epoch in range (args.num_epoch):
            correct = 0
            total_num = 0
            optimizer.zero_grad()
            for e1_index,e2_index,label,sentence_mask in train_iter:
                loss, output = model([e1_index,e2_index],label,sentence_mask.squeeze(2),features)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(output.data, 1)
                correct += predicted.data.eq(label.data).cpu().sum()
                total_num+=label.shape[0]
            loss = loss.detach().cpu()
            # torch.save(model,str(epoch)+'.pt')
            print("epoch ", str(epoch),"loss:",loss.mean().numpy().tolist(),"    acc:", correct.cpu().numpy().tolist()/total_num)
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
                loss, output = model([e1_index,e2_index],label,sentence_mask.squeeze(2),features)
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
