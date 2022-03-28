#!/bin/bash
pip install terminaltables
mkdir -p output
rm -r results/images
rm -r results/videos
TIME=`date +%Y-%b-%d-%H-%M`
python3 detect.py --weight=weights/$(ls -t weights | grep best | grep iee101_dataset | head -1) --image=data/dfki_dataset/ir/ --visual_thre=0.5
apt update -y
apt install -y ffmpeg zip
ffmpeg -r 60 -f image2 -s 512x512 -i results/images/%05d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p output/yolact_$TIME.mp4
zip -r -j output/yolact_$TIME.zip results/images/*.png output/yolact_$TIME.mp4
