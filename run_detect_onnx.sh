#!/bin/bash
pip install terminaltables
mkdir -p output
rm -r results/images_onnx
rm -r results/videos_onnx
TIME=`date +%Y-%b-%d-%H-%M`
python3 export2onnx.py --weight=weights/$(ls -t weights | grep best | grep iee101_dataset | head -1)
python3 detect_with_onnx.py --weight=onnx_files/iee101_dataset.onnx --visual_thre=0.5
apt update -y
apt install -y ffmpeg zip
ffmpeg -r 60 -f image2 -s 512x512 -i results/images_onnx/%05d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p output/yolact_onnx_$TIME.mp4
zip -r -j output/yolact_onnx_$TIME.zip results/images_onnx/*.png output/yolact_$TIME.mp4
