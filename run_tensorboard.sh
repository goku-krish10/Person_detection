#!/bin/bash
rm tensorboard_log/iee101_dataset/*
python3 -m tensorboard.main --logdir tensorboard_log/iee101_dataset --port 6006
