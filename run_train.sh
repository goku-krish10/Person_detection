#!/bin/bash
pip install terminaltables
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=$((RANDOM)) train.py --train_bs=24 --cfg=iee101_dataset
