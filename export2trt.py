#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import tensorrt as trt

parser = argparse.ArgumentParser(description='YOLACT Detection.')
parser.add_argument('--weight', default='onnx_files/iee101_dataset.onnx', type=str)
args = parser.parse_args()

trt_name = args.weight.split('/')[-1].replace('onnx', 'trt')
trt_path = f'trt_files/best_{trt_name}'

trt_logger = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
with trt.Builder(trt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
        trt.OnnxParser(network, trt_logger) as parser:

    with open(args.weight, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))

        # One of the functions of the builder is to search through its catalog of CUDA kernels for the fastest
        # implementation available, and thus it is necessary to use the same GPU for building like that on which
        # the optimized engine will run.
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)

        builder.fp16_mode = True
        builder.strict_type_constraints = True

        config.max_workspace_size = 1 << 20

        # Serialize the model to a modelstream:
        with builder.build_engine(network, config) as engine:
            with open(trt_path, 'wb') as f:
                f.write(engine.serialize())

print(f'Export succeed, saved as {trt_path}.')
