from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
# import torch

from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert

import logging
logging.basicConfig(level=logging.INFO)
import torch

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)

#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--tf_checkpoint_path",
                        default = './2/model.ckpt-280200.data-00000-of-00001',
                        type = str,
                        help = "Path to the TensorFlow checkpoint path.")
    parser.add_argument("--bert_config_file",
                        default = './2/bert_config.json',
                        type = str,
                        help = "The config json file corresponding to the pre-trained BERT model. \n"
                               "This specifies the model architecture.")
    parser.add_argument("--pytorch_dump_path",
                        default = './2/pytorch_model.bin',
                        type = str,
                        help = "Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path,
                                     args.bert_config_file,
                                     args.pytorch_dump_path)
# urllib3, pyparsing, importlib-resources, idna, charset-normalizer, requests, regex, pyyaml, packaging, joblib, filelock, click, tokenizers, sacremoses, huggingface-hub, dataclasses, transformers