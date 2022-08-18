from model_v2 import ErnieSageForLinkPrediction
import pgl
import paddle
import paddle.nn as nn
import numpy as np
from paddlenlp.transformers import ErniePretrainedModel
from paddlenlp.transformers import ErnieModel
from erniesage_model.encoder import Encoder
from erniesage_model.loss import LossFactory
__all__ = ["ErnieSageForRePrediction"]


class ErnieSageForRePrediction(ErnieSageForLinkPrediction):

    def __init__(self):
        super(ErnieSageForLinkPrediction, self).__init__()
        self.config = config
        self.ernie_graph = ernie
        self.ernie_sentence = ErnieModel.from_pretrained(config.infer_model)
        self.encoder = Encoder.factory(self.config_file, self.ernie_graph)
        self.loss_func = LossFactory(self.config_file)
        self.fc_1 = paddle.nn.Linear(1024, config.hidden_size)
        self.fc_last=paddle.nn.Linear(config.hidden_size*3,config.num_class)
    def forward(self, graphs,term_ids, data):
        """Forward function of link prediction task.

        Args:
            graphs (Graph List): the Graph list.
            datas (Tensor List): other input of the model.

        Returns:
            Tensor: loss and output tensors.
        """
        inputs_e1 = paddle.to_tensor(np.array(np.array(data)[:, 0]),dtype="int64")
        inputs_e2 = paddle.to_tensor(np.array(np.array(data)[:, 1]),dtype="int64")
        real_label = paddle.to_tensor(np.array([i for i in np.array(data)[:, 2]]))
        setence_ids = paddle.to_tensor(np.array([np.array(i) for i in np.array(data)[:, 3]]))
        # encoder model
        outputs = self.encoder([graphs], term_ids,
                               [inputs_e1, inputs_e2])
        _,sentence_feat=self.ernie_sentence(setence_ids)
        sentence_out=self.fc_1(sentence_feat)
        logits = paddle.concat([outputs[0],outputs[1],sentence_out], axis=-1)
        outputs =  self.fc_last(logits)
        loss =self.loss_func(outputs,real_label)
        return loss,outputs

class newa(ErnieSageForLinkPrediction):
    def __init__(self):
        super.__init__(self, newa)
        self.config= cacf
        self.enrie_1= ErnieModel

    def forward(self):
        inputs_e1 = paddle.to_tensor(np.array(np.array(data)[:, 0]),dtype="int64")
        inputs_e2 = paddle.to_tensor(np.array(np.array(data)[:, 1]),dtype="int64")
        out = self.enrie_1(afaf)
        real_label = paddle.to_tensor(np.array([i for i in np.array(data)[:, 2]]))
        setence_ids = paddle.to_tensor(np.array([np.array(i) for i in np.array(data)[:, 3]]))
        # encoder model
        outputs = self.encoder([graphs], term_ids,
                               [inputs_e1, inputs_e2])
        _,sentence_feat=self.ernie_sentence(setence_ids)
        sentence_out=self.fc_1(sentence_feat)
        logits = paddle.concat([outputs[0],outputs[1],sentence_out], axis=-1)
        outputs =  self.fc_last(logits)
        loss =self.loss_func(outputs,)
a= ErnieSageForLinkPrediction()

