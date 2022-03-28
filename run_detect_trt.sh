#!/bin/bash
TENSORRT_PATH="/netscratch/anisimov/TensorRT-7.2.2.3"
export PATH=$PATH:$TENSORRT_PATH/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TENSORRT_PATH/lib/
pip install $TENSORRT_PATH/python/tensorrt-7.2.2.3-cp36-none-linux_x86_64.whl
pip install $TENSORRT_PATH/uff/uff-0.6.9-py2.py3-none-any.whl
pip install $TENSORRT_PATH/graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl
pip install terminaltables pycuda
mkdir -p output
rm -r results/images_tensorrt
rm -r results/videos_tensorrt
TIME=`date +%Y-%b-%d-%H-%M`
python3 export2onnx.py --weight=weights/$(ls -t weights | grep best | grep iee101_dataset | head -1)
python3 export2trt.py
python3 detect_with_trt.py
apt update -y
apt install -y ffmpeg zip
ffmpeg -r 60 -f image2 -s 512x512 -i results/images_tensorrt/%05d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p output/yolact_tensorrt_$TIME.mp4
zip -r -j output/yolact_tensorrt_$TIME.zip results/images_tensorrt/*.png output/yolact_tensorrt_$TIME.mp4
