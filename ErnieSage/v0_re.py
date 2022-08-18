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

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import io
import random
import time
import argparse
from functools import partial

import numpy as np
import yaml
import paddle
import pgl
from easydict import EasyDict as edict
from paddlenlp.transformers import ErnieTokenizer, ErnieTinyTokenizer
from paddlenlp.utils.log import logger
from ordered_set import OrderedSet
from collections import namedtuple

from models.pretrain_model_loader import PretrainedModelLoader
from ernie import ErnieModel
import ernie
from erniesage_model import ErnieSageForLinkPrediction
# from data import TrainData, PredictData, GraphDataLoader, batch_fn

MODEL_CLASSES = {
    "ernie-tiny": (ErnieSageForLinkPrediction, ErnieTinyTokenizer),
    "ernie-1.0": (ErnieSageForLinkPrediction, ErnieTokenizer),
}


def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    paddle.seed(config.seed)

def load_graph_data(config):
    ent_set, rel_set = OrderedSet(), OrderedSet()
    for line in open(config.graph_data):
        # print(line)
        sub, obj, rel,_ = map(str.lower, line.strip().split('\t'))
        ent_set.add(sub)
        rel_set.add(rel)
        ent_set.add(obj)

    ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
    rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
    # if config.mode=='train':
    num_ent = len(ent2id)
    edge = []
    for line in open(config.graph_data):
        sub, obj, rel,_ = map(str.lower, line.strip().split('\t'))
        sub, rel, obj = ent2id[sub], rel2id[rel], ent2id[obj]
        edge.append((sub, obj))
    return num_ent, len(edge), edge, ent_set, ent2id, rel2id

def load_train_test_data(file_path, entity_map, rel_map):
    Dataset = namedtuple("Dataset",
                         ["e1_index", "e2_index", "label"])
    train_e1_index = []
    train_e2_index = []
    train_label = []
    for line in open(file_path):
        sub, obj, rel,_ = map(str.lower, line.strip().split('\t'))
        sub_index, rel_index, obj_index = entity_map[sub], rel_map[rel], entity_map[obj]
        train_e1_index.append(sub_index)
        train_e2_index.append(obj_index)
        train_label.append(rel_index)
    dataset = Dataset(
        e1_index=train_e1_index,
        e2_index=train_e2_index,
        label=train_label
    )
    return dataset

def dump_node_feat(config, entity_set):
    def term2id(string, tokenizer, max_seqlen):
        tokens = tokenizer.tokenize(string)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        ids = ids[:max_seqlen - 1]
        ids = ids + [tokenizer.sep_id]  # ids + [sep]
        ids = ids + [tokenizer.pad_id] * (max_seqlen - len(ids))
        return ids

    print("stat tokenize")
    tokenizer = ernie.tokenizing_ernie.ErnieTinyTokenizer.from_pretrained(config.model_name_or_path)
    term_ids = [
        partial(
            term2id, tokenizer=tokenizer, max_seqlen=config.max_seqlen)(s)
        for s in entity_set
    ]
    return term_ids, tokenizer,tokenizer.cls_id

def do_train(config):
    paddle.set_device(config.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(config)

    num_nodes, num_edges, edges, entity_set, ent2id, rel2id \
        = load_graph_data(config)
    config.num_class=len(rel2id)
    train_dataset = load_train_test_data(config.train_data, ent2id, rel2id)
    node_feature, tokenizer,config.cls_id = dump_node_feat(config, ent2id)
    graph = pgl.Graph(num_nodes=num_nodes, edges=edges, node_feat={"feature": np.array(node_feature).astype(np.float32)})
    graph.tensor()
    def train_reader():
        for index in range(len(train_dataset.label)):
            yield [train_dataset.e1_index[index], train_dataset.e2_index[index],
                   train_dataset.label[index]]

    train_batch_dataset = paddle.batch(train_reader, batch_size=config.batch_size)

    model_class ,_= MODEL_CLASSES[config.ernie_name]
    # ernie_model = ErnieModel.from_pretrained(config.model_name_or_path)

    model = model_class.from_pretrained(
        "ernie-tiny", config=config)
    model = paddle.DataParallel(model)

    optimizer = paddle.optimizer.SGD(
        learning_rate=config.lr, parameters=model.parameters())

    rank = paddle.distributed.get_rank()
    global_step = 0
    tic_train = time.time()
    node_feat=paddle.to_tensor(node_feature)
    for epoch in range(config.epoch):
        for step, train_iter in enumerate(train_batch_dataset()):
            global_step += 1
            loss, outputs = model(graph, node_feat, train_iter)

            if global_step % config.log_per_step == 0:
                logger.info(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss,
                       config.log_per_step /((time.time()) - tic_train)))
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if global_step % config.save_per_step == 0:
                if rank == 0:
                    output_dir = os.path.join(config.output_path,
                                              "model_%d" % global_step)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model._layers.save_pretrained(output_dir)
    if rank == 0:
        output_dir = os.path.join(config.output_path, "last")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model._layers.save_pretrained(output_dir)


def tostr(data_array):
    return " ".join(["%.5lf" % d for d in data_array])


@paddle.no_grad()
def do_predict(config):
    from sklearn.metrics import confusion_matrix
    paddle.set_device(config.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(config)

    num_nodes, num_edges, edges, entity_set, ent2id, rel2id \
        = load_graph_data(config)
    config.num_class = len(rel2id)
    id2rel={}
    for i,rel in enumerate(rel2id):
        id2rel[i]=rel
    print(id2rel)
    infer_dataset = load_train_test_data(config.test_data, ent2id, rel2id)
    node_feature, tokenizer, config.cls_id = dump_node_feat(config, ent2id)
    # print(node_feature)
    graph = pgl.Graph(num_nodes=num_nodes, edges=edges,
                      node_feat={"feature": np.array(node_feature).astype(np.float32)})
    graph.tensor()
    node_feat=paddle.to_tensor(node_feature)
    def infer_reader():
        for index in range(len(infer_dataset.label)):
            yield [infer_dataset.e1_index[index], infer_dataset.e2_index[index],
                   infer_dataset.label[index]]

    infer_batch_dataset = paddle.batch(infer_reader, batch_size=config.batch_size)

    model_class, _ = MODEL_CLASSES[config.ernie_name]

    model = model_class.from_pretrained(config.infer_model, config=config)
    model = paddle.DataParallel(model)

    trainer_id = paddle.distributed.get_rank()

    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)
    # fout = io.open(
    #     "%s/part-%s" % (config.output_path, trainer_id), "w", encoding="utf8")

    global_step = 0
    epoch = 0
    tic_train = time.time()
    model.eval()
    true_prenum = 0
    num=0
    pre_label_list = []
    real_label_list = []
    number = [0,0,0,0,0,0]
    for step, infer_iter in  enumerate(infer_batch_dataset()):
        global_step += 1
        loss, outputs = model(graph, node_feat, infer_iter)
        for i,o in enumerate(outputs):
            pre_label=paddle.argmax(o)
            number[list(np.array(pre_label))[0]]+=1
            pre_label_list.append(id2rel[list(np.array(pre_label))[0]])
            real_label=infer_iter[i][2]
            real_label_list.append(id2rel[real_label])
            if pre_label==real_label:
                true_prenum+=1
            num+=1
        print(true_prenum/num)
        if global_step % config.log_per_step == 0:
            logger.info(
                "predict step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                % (global_step, epoch, step, loss,
                   config.log_per_step / (time.time() - tic_train)))
            tic_train = time.time()

    result=confusion_matrix(real_label_list,pre_label_list)
    for i,pre in enumerate(pre_label_list):
        print(pre)
    print(result)
    print(number)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    # parser.add_argument("--conf", type=str, default="./config/erniesage_link_prediction.yaml")
    parser.add_argument("--conf", type=str, default="./config/erniesage_link_prediction.yaml")
    parser.add_argument("--do_predict", action='store_true', default=False)
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    ernie_cfg_dict, ernie_param_path = PretrainedModelLoader.from_pretrained(
        config.model_name_or_path)
    if "ernie_config" not in config:
        config.ernie_config = ernie_cfg_dict

    assert config.device in [
        "gpu", "cpu"
    ], "Device should be gpu/cpu, but got %s." % config.device
    logger.info(config)

    if args.do_predict:
        do_predict(config)
    else:
        do_train(config)
# python -m paddle.distributed.launch --gpus "3" v0_re.py --conf ./config/erniesage_link_prediction.yaml --do_predict

# Traceback (most recent call last):
#   File "relation_prediction.py", line 250, in <module>
#     do_predict(config)
#   File "/home/baichuanyang/.local/lib/python3.7/site-packages/decorator.py", line 232, in fun
#     return caller(func, *(extras + args), **kw)
#   File "/opt/anaconda3/envs/paddle/lib/python3.7/site-packages/paddle/fluid/dygraph/base.py", line 351, in _decorate_function
#     return func(*args, **kwargs)
#   File "relation_prediction.py", line 197, in do_predict
#     tokenizer = tokenizer_class.from_pretrained(config.model_name_or_path)
#   File "/opt/anaconda3/envs/paddle/lib/python3.7/site-packages/paddlenlp/transformers/tokenizer_utils.py", line 982, in from_pretrained
#     file_path, default_root)
#   File "/opt/anaconda3/envs/paddle/lib/python3.7/site-packages/paddlenlp/utils/downloader.py", line 155, in get_path_from_url
#     assert is_url(url), "downloading from {} not a url".format(url)
# AssertionError: downloading from /data/baichuanyang/project/mt/model/model-ernie_tiny.1/tokenizer_config.json not a url
