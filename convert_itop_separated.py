import h5py
import os
import numpy as np
import cv2
import argparse
import pathlib

from convert_common import *

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Conversion of ITOP annotations to COCO format')
    parser.add_argument('--input_file_depth', dest='input_file_depth',
                        help='H5 file for depth',
                        default=None, type=str)    
    parser.add_argument('--input_file_labels', dest='input_file_labels',
                        help='H5 file for labels',
                        default=None, type=str)
    parser.add_argument('--output_path', dest='output_path',
                        help='Path for saving',
                        default=None, type=str)

    args = parser.parse_args()
    return args


args = parse_args()
input_file_depth = args.input_file_depth
input_file_labels = args.input_file_labels
output_path = args.output_path   
latest_number = args.latest_number

input_file_depth = input_file_depth.replace("\\","/")
input_file_depth = input_file_depth.replace("//","/")

input_file_labels = input_file_labels.replace("\\","/")
input_file_labels = input_file_labels.replace("//","/")
            
output_path = output_path.replace("\\","/")
output_path = output_path.replace("//","/")         
            
if (os.path.exists(input_file_depth) == False):
    print("File does not exist: (",input_file_depth,")!")
    exit()
                
if (os.path.exists(input_file_labels) == False):
    print("File does not exist: (",input_file_labels,")!")
    exit()

depth_postfix = "depth/"
segm_postfix = "segm/"
filt_postfix = "filt/"

depth_path = output_path + depth_postfix
segm_path = output_path + segm_postfix
filt_path = output_path + filt_postfix

pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(depth_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(segm_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(filt_path).mkdir(parents=True, exist_ok=True)
        
f_depth = h5py.File(input_file_depth, 'r')
f_labels = h5py.File(input_file_labels, 'r')

data, ids = np.asarray(f_depth.get('data')), np.asarray(f_labels.get('id'))
id, is_valid, segmentation = np.asarray(f_labels.get('id')), np.asarray(f_labels.get('is_valid')), np.asarray(f_labels.get('segmentation')) 
len_zfill = len(str(data.shape[0])) + 1

count = 0
if os.path.isfile(latest_number):
    with open(latest_number, 'r') as file:
        count = int(file.read())

print(data.shape, ids.shape)
for ind in range(len(data)):
    if (is_valid[ind] == 1):
        im_depth = data[ind]
        im_segm = segmentation[ind] + 1

        max_value = np.max(im_depth)
        im_depth *= (255.0/max_value)
        im_depth = im_depth.astype(np.uint8)
        im_depth = 255 - im_depth
        im_depth[im_depth == 255] = 0

        im_segm[im_segm == -1] = 0
        im_segm = im_segm.astype(np.uint8)
        im_segm = cv2.threshold(im_segm, 0, 255, cv2.THRESH_BINARY)[1]

        im_filt = cv2.bitwise_and(im_depth, im_segm)

        im_depth = cv2.resize(im_depth, (544, 408))
        im_segm = cv2.resize(im_segm, (544, 408))
        im_filt = cv2.resize(im_filt, (544, 408))

        im_depth = cv2.medianBlur(im_depth,5)
        im_segm = cv2.medianBlur(im_segm,5)
        im_filt = cv2.medianBlur(im_filt,5)

        im_depth = cv2.copyMakeBorder(
            im_depth, 
            68, 68, 0, 0, 
            cv2.BORDER_CONSTANT, 0
        )
        im_segm = cv2.copyMakeBorder(
            im_segm, 
            68, 68, 0, 0, 
            cv2.BORDER_CONSTANT, 0
        )        
        im_filt = cv2.copyMakeBorder(
            im_filt, 
            68, 68, 0, 0, 
            cv2.BORDER_CONSTANT, 0
        )

        fname_depth = str(count).zfill(len_zfill) + ".png"
        fname_segm = str(count).zfill(len_zfill) + ".png"
        fname_filt = str(count).zfill(len_zfill) + ".png"
        cv2.imwrite(depth_path + fname_depth, im_depth)
        cv2.imwrite(segm_path + fname_segm, im_segm)
        cv2.imwrite(filt_path + fname_filt, im_filt)
        count += 1

with open(latest_number, 'w') as file:
  file.write('%d' % count)
