#!/bin/bash
TENSORRT_PATH="/netscratch/anisimov/TensorRT-7.2.2.3"
export PATH=$PATH:$TENSORRT_PATH/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TENSORRT_PATH/lib/
