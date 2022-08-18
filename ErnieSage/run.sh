#!/bin/bash
#cd cephfs/data/baichuanyang/project/verniesage3/
python -m paddle.distributed.launch --gpus "2" relation_prediction_1.py --conf ./config/re_erniesage_1.yaml
#python -m paddle.distributed.launch --gpus "0" v0_re.py --conf ./config/erniesage_link_prediction.yaml
#nohup /home/hadoop-aipnlp/cephfs/data/baichuanyang/project/verniesage3/run.sh  > /home/hadoop-aipnlp/cephfs/data/baichuanyang/project/verniesage3/run.log 2>&1 &