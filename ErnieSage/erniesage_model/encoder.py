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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from erniesage_model.conv import GraphSageConv, ErnieSageV2Conv


class Encoder(nn.Layer):
    """ Base class 
    Chose different type ErnieSage class.
    """

    def __init__(self, config):
        """init function

        Args:
            config (Dict): all configs.
        """
        super(Encoder, self).__init__()
        self.config = config
        # Don't add ernie to self, oterwise, there will be more copies of ernie weights 
        # self.ernie = ernie 

    @classmethod
    def factory(cls, config, ernie):
        """Classmethod for ernie sage model.

        Args:
            config (Dict): all configs.
            ernie (nn.Layer): the ernie model.

        Raises:
            ValueError: Invalid ernie sage model type.

        Returns:
            Class: real model class.
        """
        model_type = config.model_type
        if model_type == "ErnieSageV2":
            return ErnieSageV2Encoder(config, ernie)
        if model_type == "ErnieSageV3":
            return ErnieSageV3Encoder(config, ernie)
        else:
            raise ValueError("Invalid ernie sage model type")

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class ErnieSageV2Encoder(Encoder):
    def __init__(self, config, ernie):
        """ Ernie sage v2 encoder

        Args:
            config (Dict): all config.
            ernie (nn.Layer): the ernie model.
        """
        super(ErnieSageV2Encoder, self).__init__(config)
        # Don't add ernie to self, oterwise, there will be more copies of ernie weights 
        # self.ernie = ernie
        self.convs = nn.LayerList()
        initializer = None
        fc_lr = self.config.lr / 0.001
        erniesage_conv = ErnieSageV2Conv(
            ernie,
            ernie.config["hidden_size"],
            self.config.hidden_size,
            learning_rate=fc_lr,
            cls_token_id=self.config.cls_id,
            aggr_func="sum")
        self.convs.append(erniesage_conv)
        for i in range(1, self.config.num_layers):
            layer = GraphSageConv(
                self.config.hidden_size,
                self.config.hidden_size,
                learning_rate=fc_lr,
                aggr_func="sum")
            self.convs.append(layer)

        if self.config.final_fc:
            self.linear = nn.Linear(
                self.config.hidden_size,
                self.config.hidden_size,
                weight_attr=paddle.ParamAttr(learning_rate=fc_lr))

    def take_final_feature(self, feature, index):
        """Gather the final feature.

        Args:
            feature (Tensor): the total featue tensor.
            index (Tensor): the index to gather.

        Returns:
            Tensor: final result tensor.
        """
        feat = paddle.gather(feature, index)
        if self.config.final_fc:
            feat = self.linear(feat)
        if self.config.final_l2_norm:
            feat = F.normalize(feat, axis=1)
        return feat

    def forward(self, graphs, term_ids, inputs):
        """ forward train function of the model.

        Args:
            graphs (Graph List): list of graph tensors.
            inputs (Tensor List): list of input tensors.

        Returns:
            Tensor List: list of final feature tensors.
        """
        # term_ids for ErnieSageConv is the raw feature.
        feature = term_ids
        for i in range(len(graphs), self.config.num_layers):
            graphs.append(graphs[0])
        for i in range(0, self.config.num_layers):
            if i == self.config.num_layers - 1 and i != 0:
                act = None
            else:
                act = "leaky_relu"
            feature = self.convs[i](graphs[i], feature, act)

        final_feats = [self.take_final_feature(feature, x) for x in inputs]
        return final_feats

class ErnieSageV3Encoder(Encoder):
    """ ErnieSage (abbreviation of ERNIE SAmple aggreGatE), a model proposed by the PGL team.
    ErnieSageV2: Ernie is applied to the EDGE of the text graph.
    """

    def __init__(self,config,ernie,aggr_func="sum"):
        """ErnieSageV2: Ernie is applied to the EDGE of the text graph.

        Args:
            ernie (nn.Layer): the ernie model.
            input_size (int): input size of feature tensor.
            hidden_size (int): hidden size of the Conv layers.
            learning_rate (float): learning rate.
            aggr_func (str): aggregate function. 'sum', 'mean', 'max' avaliable.
        """
        super(ErnieSageV3Encoder, self).__init__(config)
        assert aggr_func in ["sum", "mean", "max", "min"], \
            "Only support 'sum', 'mean', 'max', 'min' built-in receive function."
        self.aggr_func = "reduce_%s" % aggr_func
        # self.cls_token_id = self.config.cls_id
        self.ernie_liner = nn.Linear(
            ernie.config["hidden_size"],
            self.config.hidden_size,
            weight_attr=paddle.ParamAttr(learning_rate=config.lr/0.01))
        self.ernie = ernie

    def concat_aggregate(self, gw, feature, name):
        def ernie_recv(message):
            """doc"""
            # num_neighbor = self.config.samples[0]
            # while len(feat._msg.loaded_nfeat)<num_neighbor:
            #     feat=paddle.concat(feat,paddle.zeros([self.config.max_seqlen]))
            # # pad_value = paddle.zeros([1], "int64")
            # # out, _ = L.sequence_pad(
            # #     feat, pad_value=pad_value, maxlen=num_neighbor)
            # out = paddle.reshape(feat, [0, self.config.max_seqlen * num_neighbor])
            # return out
            return getattr(message, self.aggr_func)(message["msg"])
        def _send_func(src_feat, dst_feat, edge_feat):
            return {"msg": src_feat["h"]}
        msg = gw.send(_send_func, src_feat={"h": feature})
        # msg = gw.send(lambda s, d, e: s["h"], node_feat={"h":feature})
        neigh_feature = gw.recv(ernie_recv,msg)
        neigh_feature = paddle.cast(neigh_feature, "int64")

        cls = paddle.full(shape=[self.config.batchsize],fill_value=self.config.cls_id)
        # insert cls, pop last
        term_ids = paddle.concat([cls, feature[:, :-1], neigh_feature], 1)
        term_ids.stop_gradient = True
        return term_ids

    def take_final_feature(self, feature, index, name):
        """take final feature"""
        term_ids = paddle.gather(feature, index)

        ernie_config = self.config.ernie_config
        self.slot_seqlen = self.config.max_seqlen
        position_ids = self._build_position_ids(term_ids)
        sent_ids = self._build_sentence_ids(term_ids)
        feature, _ = self.ernie(term_ids, sent_ids, position_ids)

        if self.config.final_fc:
            feature = self.ernie_liner(feature)

        if self.config.final_l2_norm:
            feature = paddle.nn.functional.normalize(feature, axis=1)
        return feature

    def _build_position_ids(self, src_ids):
        src_shape = paddle.reshape(src_ids)
        src_seqlen = src_shape[1]
        src_batch = src_shape[0]

        slot_seqlen = self.slot_seqlen

        num_b = (src_seqlen / slot_seqlen) - 1
        a_position_ids = paddle.reshape(
            paddle.arange(
                0, slot_seqlen, 1, dtype='int32'), [1, slot_seqlen])  # [1, slot_seqlen]
        a_position_ids = paddle.expand(a_position_ids,
                                       [src_batch, 1])  # [B, slot_seqlen]

        input_mask = paddle.cast(src_ids[:,:slot_seqlen] == 0, "int32")  # assume pad id == 0 [B, slot_seqlen, 1]
        a_pad_len = paddle.mean(input_mask, 1)  # [B, 1]

        b_position_ids = paddle.reshape(
            paddle.arange(
                slot_seqlen, 2 * slot_seqlen, 1, dtype='int32'),
            [1, slot_seqlen])  # [1, slot_seqlen]
        b_position_ids = paddle.expand(
            b_position_ids,
            [src_batch, num_b])  # [B, slot_seqlen * num_b]
        b_position_ids = b_position_ids - a_pad_len  # [B, slot_seqlen * num_b]

        position_ids = paddle.concat([a_position_ids, b_position_ids], 1)
        position_ids = paddle.cast(position_ids, 'int64')
        position_ids.stop_gradient = True
        return position_ids

    def _build_sentence_ids(self, src_ids):
        src_shape = paddle.reshape(src_ids)
        src_seqlen = src_shape[1]
        src_batch = src_shape[0]
        slot_seqlen = self.slot_seqlen

        zeros = paddle.zeros([src_batch, slot_seqlen], "int64")
        ones = paddle.ones([src_batch, src_seqlen - slot_seqlen], "int64")
        sentence_ids = paddle.concat([zeros, ones], 1)
        sentence_ids.stop_gradient = True
        return sentence_ids

    def ernie_send(self, src_feat, dst_feat, edge_feat):
        """ Apply ernie model on the edge.

        Args:
            src_feat (Tensor Dict): src feature tensor dict.
            dst_feat (Tensor Dict): dst feature tensor dict.
            edge_feat (Tensor Dict): edge feature tensor dict.

        Returns:
            Tensor Dict: tensor dict which use 'msg' as the key.
        """
        # input_ids
        cls = paddle.full(
            shape=[src_feat["term_ids"].shape[0], 1],
            dtype="int64",
            fill_value=self.config.cls_id)
        src_ids = paddle.concat([cls, src_feat["term_ids"]], 1)

        dst_ids = dst_feat["term_ids"]

        # sent_ids
        sent_ids = paddle.concat(
            [paddle.zeros_like(src_ids), paddle.ones_like(dst_ids)], 1)
        term_ids = paddle.concat([src_ids, dst_ids], 1)

        # build position_ids
        input_mask = paddle.cast(term_ids > 0, "int64")
        position_ids = paddle.cumsum(input_mask, axis=1) - 1

        outputs = self.ernie(term_ids, sent_ids, position_ids)
        feature = outputs[1]
        # feature = self.neigh_linear(feature)
        return {"msg": feature}

    def send_recv(self, graph, term_ids):
        """Message Passing of erniesage v2.

        Args:
            graph (Graph): the Graph object.
            feature (Tensor): the node feature tensor.

        Returns:
            Tensor: the self and neighbor feature tensors.
        """

        def _recv_func(message):
            return getattr(message, self.aggr_func)(message["msg"])

        msg = graph.send(self.ernie_send, node_feat={"term_ids": term_ids})
        neigh_feature = graph.recv(reduce_func=_recv_func, msg=msg)

        cls = paddle.full(
            shape=[term_ids.shape[0], 1],
            dtype="int64",
            fill_value=self.config.cls_id)
        term_ids = paddle.concat([cls, term_ids], 1)
        term_ids.stop_gradient = True
        outputs = self.ernie(term_ids, paddle.zeros_like(term_ids))
        self_feature = outputs[1]
        # self_feature = self.self_linear(self_feature)
        return self_feature, neigh_feature

    def forward(self, graph, term_ids, act='relu'):

        # feature = graph[0].node_feat["term_ids"]
        feature=term_ids
        feature = self.concat_aggregate(graph[0], feature, "erniesage_v3_0")

        final_feats = [
            self.take_final_feature(feature, i, "final_fc") for i in term_ids
        ]
        return final_feats