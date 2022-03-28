import h5py
import os
import numpy as np
import cv2
import argparse
import pathlib

from convert_common import add_annotation
from convert_common import draw_overlay_image
from convert_common import generate_bbox_segmentation
from convert_common import save_json
from convert_common import generate_surface_normals_new


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Conversion of ITOP \
                                     annotations to COCO format')
    parser.add_argument('--input_path', dest='input_path',
                        help='Path for input files with H5 ITOP files',
                        default="C:/pic/vizta/src/itop/", type=str)
    parser.add_argument('--output_path', dest='output_path',
                        help='Path for saving',
                        default="C:/pic/vizta/dst/itop/", type=str)
    parser.add_argument('--coef_train_val', dest='coef_train_val',
                        help='Coefficient for training/validation separations',
                        default=7, type=int)
    parser.add_argument('--depth_map_coloring',
                        dest='depth_map_coloring',
                        help='Activate it for artificially coloring depth map',
                        default=False,
                        type=bool)
    parser.add_argument('--depth_map_normals',
                        dest='depth_map_normals',
                        help='Activate it for using surface normals',
                        default=False,
                        type=bool)
    parser.add_argument('--save_overlay',
                        dest='save_overlay',
                        help='Saving the overlay image (with drawed masks)',
                        default=False,
                        type=bool)

    args = parser.parse_args()
    return args


args = parse_args()

if (args.depth_map_normals and args.depth_map_coloring):
    args.depth_map_coloring = False

input_path = args.input_path
output_path = args.output_path
coef_train_val = args.coef_train_val
depth_map_coloring = args.depth_map_coloring
depth_map_normals = args.depth_map_normals
save_overlay = args.save_overlay

input_path = input_path.replace("\\", "/")
input_path = input_path.replace("//", "/")

output_path = output_path.replace("\\", "/")
output_path = output_path.replace("//", "/")

if (os.path.exists(input_path) is False):
    print("Folder does not exist: (", input_path, ")!")
    exit()

images_postfix = "depth/"
images_filt_postfix = "finetuning/"
annotations_postfix = "annotations/"
masks_depth_postfix = "masks/"
overlay_boxes_postfix = "overlay/"

images_path = output_path + images_postfix
images_filt_path = output_path + images_filt_postfix
images_filt_train_person_path = images_filt_path + "train/person/"
images_filt_val_person_path = images_filt_path + "val/person/"
annotations_path = output_path + annotations_postfix
masks_depth_path = output_path + masks_depth_postfix
overlay_boxes_path = output_path + overlay_boxes_postfix

pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(images_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(images_filt_train_person_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(images_filt_val_person_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(annotations_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(masks_depth_path).mkdir(parents=True, exist_ok=True)

if (save_overlay):
    pathlib.Path(overlay_boxes_path).mkdir(parents=True, exist_ok=True)

if not os.path.exists(input_path + "ITOP_top_train_depth_map.h5"):
    print(input_path + "ITOP_top_train_depth_map.h5" + "does not exist.")
    exit()

if not os.path.exists(input_path + "ITOP_top_test_depth_map.h5"):
    print(input_path + "ITOP_top_test_depth_map.h5" + "does not exist.")
    exit()

if not os.path.exists(input_path + "ITOP_top_train_labels.h5"):
    print(input_path + "ITOP_top_train_labels.h5" + "does not exist.")
    exit()

if not os.path.exists(input_path + "ITOP_top_test_labels.h5"):
    print(input_path + "ITOP_top_test_labels.h5" + "does not exist.")
    exit()

_f_depth = [h5py.File(input_path + "ITOP_top_test_depth_map.h5", 'r'),
            h5py.File(input_path + "ITOP_top_train_depth_map.h5", 'r')]
_f_labels = [h5py.File(input_path + "ITOP_top_test_labels.h5", 'r'),
             h5py.File(input_path + "ITOP_top_train_labels.h5", 'r')]

json_train = {}
json_val = {}
json_trainval = {}

json_train['images'] = []
json_val['images'] = []
json_trainval['images'] = []

json_train['annotations'] = []
json_val['annotations'] = []
json_trainval['annotations'] = []

json_train_fname = "instances_train.json"
json_val_fname = "instances_val.json"
json_trainval_fname = "instances_trainval.json"

counter_image = 0
counter_annotation = 0
dilate_size = 5
kernel = np.ones((dilate_size, dilate_size), np.uint8)

try:
    for f_depth, f_labels in zip(_f_depth, _f_labels):
        data = np.asarray(f_depth.get('data'))

        ids = np.asarray(f_labels.get('id'))
        is_valid = np.asarray(f_labels.get('is_valid'))
        segmentation = np.asarray(f_labels.get('segmentation'))

        len_zfill = len(str(data.shape[0])) + 1

        print(data.shape, ids.shape)
        for ind in range(len(data)):
            if (is_valid[ind] == 1):
                im_depth = data[ind]
                im_segm = segmentation[ind] + 1

                im_depth[im_depth < 0.25] = 0
                im_depth[im_depth > 2.88] = 0
                im_depth = im_depth - 0.24
                im_depth[im_depth < 0] = 0
                im_depth = im_depth / (2.88 - 0.25)
                im_depth = (im_depth * 255).astype(np.uint8)

                im_segm = im_segm.astype(np.uint8)
                im_segm = cv2.threshold(im_segm, 0, 255, cv2.THRESH_BINARY)[1]

                im_filt = cv2.bitwise_and(im_depth, im_segm)

                im_depth = cv2.resize(im_depth, (512, 384))
                im_segm = cv2.resize(im_segm, (512, 384))
                im_filt = cv2.resize(im_filt, (512, 384))

                im_depth = cv2.copyMakeBorder(
                    im_depth,
                    64, 64, 0, 0,
                    cv2.BORDER_CONSTANT, 0
                )
                im_segm = cv2.copyMakeBorder(
                    im_segm,
                    64, 64, 0, 0,
                    cv2.BORDER_CONSTANT, 0
                )
                im_filt = cv2.copyMakeBorder(
                    im_filt,
                    64, 64, 0, 0,
                    cv2.BORDER_CONSTANT, 0
                )

                areas = []
                bboxes = []
                segmentations = []
                segm, bbox, area = generate_bbox_segmentation(
                    im_segm)
                if ((segm is None) or (bbox is None) or (area is None)):
                    bboxes = False
                else:
                    segmentations.append(segm)
                    bboxes.append(bbox)
                    areas.append(area)

                if (bboxes is not False):
                    counter_image_zeros = str(counter_image).zfill(len_zfill)
                    fname = counter_image_zeros + ".png"

                    im_height, im_width = im_depth.shape

                    dict_annotation = {
                        "counter_image": counter_image,
                        "counter_annotation": counter_annotation,
                        "fname": "../depth/" + fname,
                        "width": im_width,
                        "height": im_height,
                        "coef_train_val": coef_train_val,
                        "num_persons": 1
                    }

                    is_val_set = (
                        (not (counter_image % coef_train_val))
                        and (counter_image != 0))
                    counter_annotation = add_annotation(
                        json_trainval, json_train, json_val,
                        segmentations, bboxes, areas,
                        dict_annotation, is_val_set)

                    if (is_val_set):
                        cv2.imwrite(
                            images_filt_val_person_path + fname, im_filt)
                    else:
                        cv2.imwrite(
                            images_filt_train_person_path + fname, im_filt)

                    if (depth_map_coloring):
                        im_depth = cv2.applyColorMap(
                            im_depth,
                            cv2.COLORMAP_JET
                        )
                    elif (depth_map_normals):
                        im_depth_normals = \
                            generate_surface_normals_new(
                                im_depth
                            )
                        im_depth = im_depth_normals.copy()

                    cv2.imwrite(images_path + fname, im_depth)
                    cv2.imwrite(masks_depth_path + fname, im_segm)
                    if (save_overlay):
                        overlay = draw_overlay_image(
                            im_depth,
                            bboxes, segmentations, True)
                        cv2.imwrite(overlay_boxes_path + fname, overlay)

                    counter_image += 1
                    print("Done with image", counter_image, "ind:", ind)

        data = []
        ids = []
        is_valid = []
        segmentation = []

except KeyboardInterrupt:
    print('Interrupted.')

save_json(annotations_path,
          json_trainval_fname, json_train_fname, json_val_fname,
          json_trainval, json_train, json_val)

print("MAX VALUE THROUGH ALL IMAGES: ", max_value_total)