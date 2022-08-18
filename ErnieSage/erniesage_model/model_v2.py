# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pgl
import paddle
import paddle.nn as nn
import numpy as np
from paddlenlp.transformers import ErniePretrainedModel
# from paddlenlp.transformers import ErnieModel
from erniesage_model.encoder import Encoder
from erniesage_model.loss import LossFactory
__all__ = ["ErnieSageForLinkPrediction"]


class ErnieSageForLinkPrediction(ErniePretrainedModel):
    """ErnieSage for link prediction task.
    """

    def __init__(self, ernie_graph,ernie_sentence,config):
        """ Model which Based on the PaddleNLP PretrainedModel

        Note: 
            1. the ernie must be the first argument.
            2. must set self.XX = ernie to load weights.
            3. the self.config keyword is taken by PretrainedModel class.

        Args:
            ernie (nn.Layer): the submodule layer of ernie model. 
            config (Dict): the config file
        """
        super(ErnieSageForLinkPrediction, self).__init__()
        self.config_file = config
        self.ernie_graph = ernie_graph
        self.ernie_sentence = ernie_sentence
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

