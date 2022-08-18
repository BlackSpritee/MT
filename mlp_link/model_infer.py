from transformers import BertModel
import torch.nn as nn
import torch
class BERT_Classifier(nn.Module):
    def __init__(self,label_num):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.1,inplace=False)
        self.fc2 = nn.Linear(768, label_num)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x, mask,label=None):

        x = self.encoder(x, attention_mask=mask)[0]
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.fc2(x)
        if label == None:
            return None,x
        else:
            return self.criterion(x,label),x

PATH="save_model/v2_entity_sentence_type_model.pt"

def map_id_rel():
    # rel2id={'synonym':0,'hypernym':1,'uncorrelated':2, 'needA':3, 'needSpecific':4, 'correlated':5,'hasA':6}
    rel2id={'synonym':0,'hypernym':1,'uncorrected':2, 'needA':3, 'needSpecific':4, 'correlated':5,'hasA':6,'suitable':7,'typical':8}
    id2rel={}
    for i in rel2id:
        id2rel[rel2id[i]]=i
    return rel2id,id2rel

def infer(model,batch_size,dataset):
    _,id_rel=map_id_rel()
    model.load_state_dict(torch.load(PATH)["state_dict"]["network"])
    if torch.cuda.is_available():
        model=model.cuda()
    model.eval()
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    result=[]
    with torch.no_grad():
        for text,mask in train_iter:
            if text.size(0)!=batch_size:
                break
            text = text.reshape(batch_size,-1)
            mask = mask.reshape(batch_size, -1)
            if torch.cuda.is_available():
                text=text.cuda()
                mask=mask.cuda()
            outputs= model(text, mask)
            loss, logits = outputs[0],outputs[1]
            logits = torch.nn.functional.softmax(logits.data, 1)
            score, predicted=torch.max(logits.data, 1)
            for index,pre_id in enumerate(predicted.cpu().numpy().tolist()) :
                result.append({"predict_rel":str(id_rel[pre_id]),"predict_score":float(score.cpu().numpy().tolist()[index])})
        print(result)
        return result