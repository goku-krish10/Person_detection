import argparse
import cv2
import fnmatch
import glob
import os
import numpy as np
import pathlib
import tarfile
import csv

from math import floor
from sys import exit
from random import randrange
from random import seed
from random import shuffle

from convert_common import add_annotation
from convert_common import draw_overlay_image
from convert_common import draw_overlay_image_new
from convert_common import item_name_vizta_convention
from convert_common import replace_slash
from convert_common import normalize
from convert_common import apply_resizing_with_borders
from convert_common import rotate_image
from convert_common import create_motion_blur
from convert_common import generate_bbox_segmentation
from convert_common import save_json
from convert_common import pencil_filter
from convert_common import generate_surface_normals_new
from convert_common import get_frame_number
from convert_common import get_timestamp
from convert_common import parse_dir
from convert_common import parse_fname
from convert_common import fill_masks
from convert_common import frame_name_dataset_conversion
from convert_common import folder_name_dataset_conversion


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Conversion of IEE/DFKI \
                                    labeling tool annotations to COCO format')

    parser.add_argument('--dataset_folder_path',
                        dest='dataset_folder_path',
                        help='Path to the folder with data \
                            (PRSX_XX_XX_WALK_Y_XX-\
                            XX_NA_250_XXXXXXXXXXXXXX_Az1_cs04)',
                        default='C:/pic/vizta/src/iee_dataset_training/',
                        type=str)

    parser.add_argument('--save_path',
                        dest='save_path',
                        help='Path for saving the data and annotations',
                        default='C:/pic/vizta/dst/iee_dataset_training/',
                        type=str)

    parser.add_argument('--area_thresh',
                        dest='area_thresh',
                        help='Area threshold (full resolution)',
                        default=500,
                        type=int)

    parser.add_argument('--coef_train_val',
                        dest='coef_train_val',
                        help='Coefficient for training/validation dataset \
                            separations',
                        default=7,
                        type=int)

    parser.add_argument('--clip_ir',
                        dest='clip_ir',
                        help='Switch for clipping IR images',
                        default=False,
                        type=bool)

    parser.add_argument('--debug_info',
                        dest='debug_info',
                        help='Activate it for printing various debugging info',
                        default=False,
                        type=bool)

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

    parser.add_argument('--dilate_masks',
                        dest='dilate_masks',
                        help='Switch for masks dilation',
                        default=True,
                        type=bool)

    parser.add_argument('--enable_augmentation',
                        dest='enable_augmentation',
                        help='Switch for data augmentation',
                        default=True,
                        type=bool)

    parser.add_argument('--enable_hull',
                        dest='enable_hull',
                        help='Switch for changing segmentation to convex hull',
                        default=True,
                        type=bool)

    parser.add_argument('--enable_motion_blur',
                        dest='enable_motion_blur',
                        help='Switch for motion blur',
                        default=False,
                        type=bool)

    parser.add_argument('--fov',
                        dest='fov',
                        help='Field of view',
                        default=120,
                        type=int)

    parser.add_argument('--generate_statistics',
                        dest='generate_statistics',
                        help='Switch for generating per-sequence statistics',
                        default=True,
                        type=bool)

    parser.add_argument('--im_height',
                        dest='im_height',
                        help='Desired image height (after rectification)',
                        default=512,
                        type=int)

    parser.add_argument('--im_width',
                        dest='im_width',
                        help='Desired image width (after rectification)',
                        default=512,
                        type=int)

    parser.add_argument('--is_8_bit_images',
                        dest='is_8_bit_images',
                        help='Switch for conversion of 16-bit images \
                            to 8-bit format',
                        default=True,
                        type=bool)

    parser.add_argument('--is_separated_folder',
                        dest='is_separated_folder',
                        help='Switch for processing the separated folder \
                            instead of subfolders',
                        default=False,
                        type=bool)

    parser.add_argument('--k_alpha',
                        dest='k_alpha',
                        help='Alpha value for the \
                            cv2.getOptimalNewCameraMatrix()',
                        default=0,
                        type=int)

    parser.add_argument('--log_ir',
                        dest='log_ir',
                        help='Switch for logarithmical IR images',
                        default=True,
                        type=bool)

    parser.add_argument('--masks_filling',
                        dest='masks_filling',
                        help='Switch for filling the masks (mismatch fix)',
                        default=False,
                        type=bool)

    parser.add_argument('--max_replacement',
                        dest='max_replacement',
                        help='Replaces UINT8_MAX with zero',
                        default=True,
                        type=bool)

    parser.add_argument('--new_annotation_remapping',
                        dest='new_annotation_remapping',
                        help='Switch for experiments with annotations',
                        default=True,
                        type=bool)

    parser.add_argument('--original_camera_matrix',
                        dest='original_camera_matrix',
                        help='Switch for estimating the camera matrix \
                            based on cv2.getOptimalNewCameraMatrix()',
                        default=False,
                        type=bool)

    parser.add_argument('--pencil_edges',
                        dest='pencil_edges',
                        help='Multiplication by result of modified pencil',
                        default=False,
                        type=bool)

    parser.add_argument('--relative_path_annotation',
                        dest='relative_path_annotation',
                        help='Enable to use relative path to image files \
                            in annotations',
                        default=True,
                        type=bool)

    parser.add_argument('--normalize_depth',
                        dest='normalize_depth',
                        help='Switch for normalizing depth values',
                        default=False,
                        type=bool)

    parser.add_argument('--save_calibration',
                        dest='save_calibration',
                        help='Saving the specific calibration file',
                        default=False,
                        type=bool)

    parser.add_argument('--save_ir',
                        dest='save_ir',
                        help='Saving the infrared image',
                        default=False,
                        type=bool)

    parser.add_argument('--save_overlay',
                        dest='save_overlay',
                        help='Saving the overlay image (with drawed masks)',
                        default=False,
                        type=bool)

    parser.add_argument('--save_masks_thresh',
                        dest='save_masks_thresh',
                        help='Saving the thresholded masks',
                        default=False,
                        type=bool)

    parser.add_argument('--shuffle_images',
                        dest='shuffle_images',
                        help='Shuffeling the data',
                        default=True,
                        type=bool)

    parser.add_argument('--ticam_format',
                        dest='ticam_format',
                        help='Saving the data in TiCam dataset format',
                        default=False,
                        type=bool)

    parser.add_argument('--val_sequences',
                        dest='val_sequences',
                        help='Using sequences for validation (instead of img)',
                        default=False,
                        type=bool)

    parser.add_argument('--zero_replacement',
                        dest='zero_replacement',
                        help='Replaces zero with UINT8_MAX/UINT16_MAX',
                        default=False,
                        type=bool)

    parser.add_argument('--low_d',
                        dest='low_d',
                        help='Low value for depth filtration',
                        default=250,
                        type=int)

    parser.add_argument('--high_d',
                        dest='high_d',
                        help='Low value for depth filtration',
                        default=2880,
                        type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if (args.depth_map_normals and args.depth_map_coloring):
        args.depth_map_coloring = False

    if (args.log_ir and args.clip_ir):
        args.clip_ir = False

    if not (args.ticam_format):
        args.generate_statistics = False

    if (args.masks_filling):
        args.save_ir = True

    if (args.ticam_format):
        args.area_thresh = 0
        args.original_camera_matrix = False
        args.depth_map_normals = False
        args.depth_map_coloring = False
        args.dilate_masks = False
        args.enable_augmentation = False
        args.enable_hull = False
        args.enable_motion_blur = False
        args.save_calibration = True
        args.normalize_depth = False

        args.generate_statistics = True

        args.clip_ir = False
        args.log_ir = False
        args.save_ir = True

    area_thresh = args.area_thresh
    coef_train_val = args.coef_train_val
    clip_ir = args.clip_ir
    dataset_folder_path = replace_slash(args.dataset_folder_path)
    debug_info = args.debug_info
    depth_map_coloring = args.depth_map_coloring
    depth_map_normals = args.depth_map_normals
    dilate_masks = args.dilate_masks
    enable_augmentation = args.enable_augmentation
    enable_hull = args.enable_hull
    enable_motion_blur = args.enable_motion_blur
    fov = args.fov
    generate_statistics = args.generate_statistics
    im_height = args.im_height
    im_width = args.im_width
    is_8_bit_images = args.is_8_bit_images
    is_separated_folder = args.is_separated_folder
    k_alpha = args.k_alpha
    log_ir = args.log_ir
    save_masks_thresh = args.save_masks_thresh
    masks_filling = args.masks_filling
    new_annotation_remapping = args.new_annotation_remapping
    normalize_depth = args.normalize_depth
    original_camera_matrix = args.original_camera_matrix
    pencil_edges = args.pencil_edges
    relative_path_annotation = args.relative_path_annotation
    save_calibration = args.save_calibration
    save_ir = args.save_ir
    save_overlay = args.save_overlay
    save_path = replace_slash(args.save_path)
    shuffle_images = args.shuffle_images
    ticam_format = args.ticam_format
    val_sequences = args.val_sequences
    zero_replacement = args.zero_replacement

    low_d = args.low_d  # low_d = 250
    high_d = args.high_d  # high_d = 2880

    print('Called with args:')
    print(args)

    if (dilate_masks):
        dilate_size = 9
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (dilate_size, dilate_size))

    # Prepare paths
    labels_prefix = '_labels/'
    images_folders = []

    if (not os.path.exists(dataset_folder_path)):
        print("Path", dataset_folder_path, "does not exist.")
        exit(1)

    if (not os.path.exists(save_path)):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    if (is_separated_folder):
        image_folder = dataset_folder_path.rsplit(labels_prefix)[0]
        images_folders.append(image_folder)
    else:
        glob_pattern = dataset_folder_path + "*" + labels_prefix

        labels_folders = glob.glob(glob_pattern)

        for label_folder in labels_folders:
            label_folder = replace_slash(label_folder)
            image_folder = label_folder.rsplit(labels_prefix)[0]
            if (not os.path.exists(image_folder)):
                print("Path", image_folder, "does not exist.")
            else:
                images_folders.append(image_folder)

    # Creating the folders for result saving

    depth_prefix = "depth"
    ir_prefix = "ir"
    masks_prefix = "masks"
    annotations_prefix = "annotations"

    depth_undist_path = save_path + depth_prefix + "/"
    ir_undist_path = save_path + ir_prefix + "/"
    masks_undist_path = save_path + masks_prefix + "/"
    annotations_path = save_path + annotations_prefix + "/"

    overlay_path = save_path + "overlay/"
    masks_thresh_path = save_path + "masks_thresh/"

    new_calibration_path = save_path + "calibration/"

    folders_undist = [
        depth_undist_path,
        masks_undist_path,
        annotations_path]

    if (save_ir):
        folders_undist.append(ir_undist_path)
    if (save_overlay):
        folders_undist.append(overlay_path)
    if (save_masks_thresh):
        folders_undist.append(masks_thresh_path)
    if (save_calibration):
        folders_undist.append(new_calibration_path)

    if (not os.path.exists(save_path)):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    for folder in folders_undist:
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    if (generate_statistics):
        timestamp = get_timestamp()
        stat_persons_filename = save_path + \
            "DataAnnotationStatistics_" + \
            timestamp + "_persons" + ".csv"
        csv_writer_stat_persons_file = open(stat_persons_filename, 'w',
                                            newline='', encoding='utf-8')
        csv_writer_stat_persons = csv.writer(
                csv_writer_stat_persons_file,
                delimiter=';')

        stat_sequences_filename = save_path + \
            "DataAnnotationStatistics_" + \
            timestamp + "_sequences" + ".csv"
        csv_writer_stat_sequences_file = open(stat_sequences_filename, 'w',
                                              newline='', encoding='utf-8')
        csv_writer_stat_sequences = csv.writer(
                csv_writer_stat_sequences_file,
                delimiter=';')

        stat_summary_filename = save_path + \
            "DataAnnotationStatistics_" + \
            timestamp + "_summary" + ".csv"
        csv_writer_stat_summary_file = open(stat_summary_filename, 'w',
                                            newline='', encoding='utf-8')
        csv_writer_stat_summary = csv.writer(
                csv_writer_stat_summary_file,
                delimiter=';')

        csv_writer_stat_sequences.writerow(["fname",
                                            "nframes",
                                            "nframes1",
                                            "nframes2",
                                            "nframesm",
                                            "nannot",
                                            "ntrunc",
                                            "ntrunc2"])

        csv_writer_stat_persons.writerow(["height",
                                          "person",
                                          "seq",
                                          "per_1",
                                          "per_1+",
                                          "per_2",
                                          "per_2+",
                                          "per_m",
                                          "frames",
                                          "length"])

        dict_sequence_sum = {
            "fname": "",
            "nframes": 0,
            "nframes1": 0,
            "nframes2": 0,
            "nframesm": 0,
            "nannot": 0,
            "ntrunc": 0,
            "ntrunc2": 0,
            "nocc": 0,
            "length": 0,
        }

        dict_global = {}

        dict_global_sum = {
            'seq': 0,
            'per_1': 0,
            'per_1+': 0,
            'per_2': 0,
            'per_2+': 0,
            'per_m': 0,
            'frames': 0,
            'length': 0,
        }

        dict_summary = {'sum': {
                'seq_total': 0,
                'per_1': 0,
                'per_1+': 0,
                'per_2': 0,
                'per_2+': 0,
                'per_m': 0,
                'diff_pers': 0,
                'frames': 0,
                'length': 0,
                'avg_frames': 0,  # frames/seq
                'avg_length': 0,  # length/seq
                'fr_': 0,
                'fr_1': 0,
                'fr_2': 0,
                'fr_m': 0,
                'ann_': 0,
                'ann_non_tr': 0,
                'ann_tr_1': 0,
                'ann_tr_2': 0
            }
        }

    json_train = {}
    json_val = {}
    json_trainval = {}

    json_train['images'] = []
    json_val['images'] = []
    json_trainval['images'] = []

    json_train['annotations'] = []
    json_val['annotations'] = []
    json_trainval['annotations'] = []

    json_train['categories'] = [
        {"supercategory": "person", "id": 1, "name": "person"},
    ]
    json_val['categories'] = [
        {"supercategory": "person", "id": 1, "name": "person"},
    ]
    json_trainval['categories'] = [
        {"supercategory": "person", "id": 1, "name": "person"},
    ]

    json_train_fname = "instances_train.json"
    json_val_fname = "instances_val.json"
    json_trainval_fname = "instances_trainval.json"

    # Augmentation parameters
    if (enable_augmentation):
        angles = list(range(0, 360-1, 30))
        scales = [1]
    else:
        angles = [0]
        scales = [1]

    motion = [5]

    angles_size = len(angles)
    scales_size = len(scales)
    motion_size = len(motion)

    counter_image = 0
    counter_annotation = 0

    seed(69)

    # For every folder in dataset
    for dataset_path in images_folders:
        dataset_dir = os.path.basename(os.path.dirname(dataset_path + "/"))

        if (val_sequences):
            is_val_set = (randrange(coef_train_val) == 0)

        labels_path = dataset_path + labels_prefix
        dataset_path = dataset_path + "/"
        calibration_path = labels_path + "calibration/"
        masks_depth_path = labels_path + "masks_depth/"

        # Checking if corresponding folders exist

        if (not os.path.exists(masks_depth_path)):
            print("Path", labels_path, "does not exist.")
            continue

        if (not os.path.exists(dataset_path)):
            print("Dataset directory does not exist: (",
                  dataset_path, ")!")
            continue

        if (not os.path.exists(calibration_path)):
            print("Calibration directory does not exist: (",
                  calibration_path, ")!")
            continue

        if (not os.path.exists(labels_path)):
            print("Labels directory does not exist: (",
                  labels_path, ")!")
            continue

        if (not os.path.exists(masks_depth_path)):
            print("Masks depth directory does not exist: (",
                  masks_depth_path, ")!")
            continue

        if (ticam_format):
            if (not os.path.exists(labels_path + "boxes_3d.csv")):
                print("3D bounding boxes file does not exist: (",
                      labels_path + "boxes_3d.csv", ")!")
                continue

        # Camera directory
        if (ticam_format):
            name_splitted = dataset_dir.split("_")
            cam_name = name_splitted[-1]
            folder_name = '_'.join(name_splitted[:-1])
            folder_name = folder_name_dataset_conversion(folder_name)

            depth_undist_path = save_path + depth_prefix + "/" + \
                cam_name + "/" + folder_name + "/"
            ir_undist_path = save_path + ir_prefix + "/" + \
                cam_name + "/" + folder_name + "/"
            masks_undist_path = save_path + masks_prefix + "/" + \
                cam_name + "/" + folder_name + "/"
            annotations_path = save_path + annotations_prefix + "/" + \
                cam_name + "/" + folder_name + "/"

            overlay_path = save_path + "overlay/" + \
                cam_name + "/" + folder_name + "/"
            masks_thresh_path = save_path + "masks_thresh/" + \
                cam_name + "/" + folder_name + "/"

            new_calibration_path = save_path + "calibration/" + \
                cam_name + "/" + folder_name + "/"

            folders_undist = [
                depth_undist_path,
                masks_undist_path,
                annotations_path]

            if (save_ir):
                folders_undist.append(ir_undist_path)
            if (save_overlay):
                folders_undist.append(overlay_path)
            if (save_masks_thresh):
                folders_undist.append(masks_thresh_path)
            if (save_calibration):
                folders_undist.append(new_calibration_path)

            for folder in folders_undist:
                pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

            # 2D, 3D CSV
            items_2d = [
                "frame",
                "label",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "area"]
            items_3d = [
                "frame",
                "label",
                "cx",
                "cy",
                "cz",
                "dx",
                "dy",
                "dz",
                "phi_x",
                "phi_y",
                "phi_z"]

            csv_reader_3d_file = open(labels_path + "boxes_3d.csv", 'r')
            csv_reader_3d = csv.reader(
                csv_reader_3d_file,
                delimiter=',')
            next(csv_reader_3d)

            csv_writer_2d_file = open(annotations_path + "boxes_2d.csv",
                                      'w', newline='', encoding='utf-8')
            csv_writer_2d = csv.writer(
                csv_writer_2d_file,
                delimiter=';')

            csv_writer_3d_file = open(annotations_path + "boxes_3d.csv",
                                      'w', newline='', encoding='utf-8')
            csv_writer_3d = csv.writer(
                csv_writer_3d_file,
                delimiter=';')

            csv_writer_2d.writerow(items_2d)
            csv_writer_3d.writerow(items_3d)

        # Checking, if classes definitions are in place
        len_files = len(fnmatch.filter(
            os.listdir(masks_depth_path), '**_classes.png'))

        if (len_files == 0):
            print("No images to annotate in", dataset_path)
            continue
        else:
            len_files *= angles_size * scales_size
            print("Annotating", len_files // (angles_size * scales_size),
                  "images from", dataset_path)

        # Loading the calibration data

        calibration_dist = calibration_path + "calib_distorted.yml"
        calibration_undist = calibration_path + "calib_undistorted.yml"

        yml_file_dist = cv2.FileStorage(
            calibration_dist, cv2.FILE_STORAGE_READ)

        k_dist = np.float32(
            yml_file_dist.getNode('k_depth').mat())
        dist_params = np.float32(
            yml_file_dist.getNode('d_depth').mat().squeeze())

        yml_file_dist.release()

        yml_file_undist = cv2.FileStorage(
            calibration_undist, cv2.FILE_STORAGE_READ)

        k_undist = np.float32(
            yml_file_undist.getNode('k_depth').mat())
        undist_params = np.float32(
            yml_file_undist.getNode('d_depth').mat().squeeze())

        r_undist = np.float32(
            yml_file_undist.getNode('r_depth').mat())
        t_undist = np.float32(
            yml_file_undist.getNode('t_depth').mat())

        yml_file_undist.release()

        common_mtx, _ = cv2.getOptimalNewCameraMatrix(
            k_dist, dist_params, (im_height, im_width), k_alpha)

        if not (original_camera_matrix):
            fov_orig = 120.0 * np.pi / 180.0
            dim_orig = 512
            f_orig = 153
            x = f_orig / (dim_orig / (2.0 * np.tan(fov_orig/2.0)))
            # x = 1.0
            div_fov = 2.0 * np.tan((fov*np.pi/180.0)/2.0)
            focal_length_x = (im_width * x) / div_fov
            focal_length_y = (im_height * x) / div_fov
            camera_center_x = im_width/2
            camera_center_y = im_height/2

            common_mtx = np.array([
                [focal_length_x, 0.0, camera_center_x],
                [0.0, focal_length_y, camera_center_y],
                [0.0, 0.0, 1.0]])

        print(str((common_mtx[0][0] + common_mtx[1][1])/2.0))

        if (save_calibration):
            yml_file_new = cv2.FileStorage(
                new_calibration_path + "calibration.yml", 
                cv2.FILE_STORAGE_WRITE)
            yml_file_new.write("K", common_mtx)
            yml_file_new.write("d", undist_params)
            yml_file_new.write("R", r_undist)
            yml_file_new.write("t", t_undist)
            yml_file_new.release()

        # Maps for original image sizes
        map1, map2 = cv2.initUndistortRectifyMap(
            k_undist, undist_params, R=np.eye(3),
            newCameraMatrix=common_mtx,
            size=(im_width, im_height),
            m1type=cv2.CV_32FC2)

        # Maps for depth and IR
        map3, map4 = cv2.initUndistortRectifyMap(
            k_dist, dist_params, R=np.eye(3),
            newCameraMatrix=common_mtx,
            size=(im_width, im_height),
            m1type=cv2.CV_32FC2)

        mask_pattern = masks_depth_path + "**_classes.png"
        list_depth_masks = glob.glob(mask_pattern, recursive=True)

        if (len(list_depth_masks) > 0):
            _mask = cv2.imread(
                    list_depth_masks[0],
                    cv2.IMREAD_ANYDEPTH)
            h, w = _mask.shape
        else:
            print("Masks list is empty!")
            break

        if (new_annotation_remapping):
            _im = np.ones((im_height, im_width), dtype=np.uint16)
            _im_pts = np.nonzero(_im)
            _im_pts = np.asarray(_im_pts).astype(np.float64).T
            _im_pts[:, [0, 1]] = _im_pts[:, [1, 0]]
            _im_pts_undist = cv2.undistortPoints(
                _im_pts,
                k_dist,
                dist_params, P=k_undist).reshape(-1, 2)
            _im_pts = _im_pts.astype(int)
            _im_pts_undist = np.around(_im_pts_undist).astype(
                                int).reshape(-1, 2)

            _joined = np.concatenate((_im_pts, _im_pts_undist),
                                     axis=1)
            _joined = _joined[_joined[:, 2] > 0]
            _joined = _joined[_joined[:, 2] < (w - 1)]
            _joined = _joined[_joined[:, 3] > 0]
            _joined = _joined[_joined[:, 3] < (h - 1)]

            _im_undist_x = np.zeros((h, w),
                                    dtype=np.float32)
            _im_undist_y = np.zeros((h, w),
                                    dtype=np.float32)
            for _p in _joined:
                _im_undist_x[_p[1], _p[0]] = _p[2]
                _im_undist_y[_p[1], _p[0]] = _p[3]
            # TODO: apply the remapping table to
            # bring the annotation to original format

        if (debug_info):
            print("Depth mask (Pre): ", mask_pattern)
            print("List of depth masks: ", list_depth_masks)

        if (generate_statistics):
            dict_sequence = {
                "fname": dataset_dir,
                "nframes": 0,
                "nframes1": 0,
                "nframes2": 0,
                "nframesm": 0,
                "nannot": 0,
                "ntrunc": 0,
                "ntrunc2": 0,
                "ntrucnnon": 0,
                "nocc": 0,
                "length": 0,
                "num_persons": 0,
                "add_object": False
            }
            dict_global_folder = parse_dir(dataset_dir)

        seq_count = 0
        start_time = None
        # if (save_calibration):
        #     list_depth_masks = []

        try:
            for item in list_depth_masks:
                item = item.replace("\\", "/")
                item = item.replace("//", "/")

                item_name = item.split(depth_prefix + '/')[1].split(
                                      '_Z_classes')[0]

                item_classes_src = item_name + "_Z_classes.png"
                item_instances_src = item_name + "_Z_instances.png"
                item_depth_src = item_name + "_Z.png"
                item_ir_src = item_name + "_IR.png"

                path_classes_src = masks_depth_path + item_classes_src
                path_instances_src = masks_depth_path + item_instances_src
                path_depth_src = dataset_path + item_depth_src
                path_ir_src = dataset_path + item_ir_src

                if(debug_info):
                    print("item_name: ", item_name)

                    print("item_classes: ", item_classes_src)
                    print("item_instances: ", item_instances_src)
                    print("item_depth: ", item_depth_src)
                    print("item_ir: ", item_ir_src)

                    print("path_classes: ", path_classes_src)
                    print("path_instances: ", path_instances_src)
                    print("path_depth: ", path_depth_src)
                    print("path_ir: ", path_ir_src)

                if (os.path.exists(path_classes_src) is False):
                    print("Class does not exist: (",
                          path_classes_src, ")!")
                    break

                if (os.path.exists(path_instances_src) is False):
                    print("Instance does not exist: (",
                          path_instances_src, ")!")
                    break

                if (os.path.exists(path_depth_src) is False):
                    print("Depth does not exist: (",
                          path_depth_src, ")!")
                    break

                if (save_ir):
                    if (os.path.exists(path_ir_src) is False):
                        print("IR does not exist: (",
                              path_ir_src, ")!")
                        break

                im_mask_classes_src = cv2.imread(path_classes_src,
                                                 cv2.IMREAD_ANYDEPTH)
                # If mask is non-empty
                if im_mask_classes_src.max() != 0:
                    # Filename conversion according to the VIZTA convention
                    frame_name = item_name_vizta_convention(item_name)
                    # Add fname here
                    frame_number = get_frame_number(frame_name)
                    frame_name = frame_name_dataset_conversion(frame_name)

                    # Instances mask, depth and IR are loaded
                    # only if the classes mask is valid

                    im_mask_instances_src = cv2.imread(
                        path_instances_src, cv2.IMREAD_ANYDEPTH)
                    im_depth_src = cv2.imread(
                        path_depth_src, cv2.IMREAD_ANYDEPTH)
                    if (save_ir):
                        im_ir_src = cv2.imread(
                            path_ir_src, cv2.IMREAD_ANYDEPTH)

                    # Look-up tables for masks, depth and IR

                    if (new_annotation_remapping):
                        _im_mask_classes_src = cv2.remap(
                            im_mask_classes_src, _im_undist_x, _im_undist_y,
                            interpolation=cv2.INTER_NEAREST)
                        _im_mask_instances_src = cv2.remap(
                            im_mask_instances_src, _im_undist_x, _im_undist_y,
                            interpolation=cv2.INTER_NEAREST)

                        im_depth_dst = cv2.remap(
                            im_depth_src, map3, map4,
                            interpolation=cv2.INTER_NEAREST)
                        im_mask_classes_dst = cv2.remap(
                            _im_mask_classes_src, map3, map4,
                            interpolation=cv2.INTER_NEAREST)
                        im_mask_instances_dst = cv2.remap(
                            _im_mask_instances_src, map3, map4,
                            interpolation=cv2.INTER_NEAREST)
                    else:
                        im_depth_dst = cv2.remap(
                            im_depth_src, map3, map4,
                            interpolation=cv2.INTER_NEAREST)
                        im_mask_classes_dst = cv2.remap(
                            im_mask_classes_src, map1, map2,
                            interpolation=cv2.INTER_NEAREST)
                        im_mask_instances_dst = cv2.remap(
                            im_mask_instances_src, map1, map2,
                            interpolation=cv2.INTER_NEAREST)

                    if (save_ir):
                        im_ir_dst = cv2.remap(
                            im_ir_src, map3, map4,
                            interpolation=cv2.INTER_LINEAR).astype(np.float32)

                    if (dilate_masks):
                        # Dilating the masks
                        im_depth_dst_thresh = cv2.threshold(
                            im_depth_dst, 0, 1, cv2.THRESH_BINARY)[1]
                        classmask_remapped = cv2.morphologyEx(
                            im_mask_classes_dst, cv2.MORPH_CLOSE,
                            dilate_kernel) * im_depth_dst_thresh
                        instancemask_remapped = cv2.morphologyEx(
                            im_mask_instances_dst, cv2.MORPH_CLOSE,
                            dilate_kernel) * im_depth_dst_thresh
                    else:
                        classmask_remapped = im_mask_classes_dst
                        instancemask_remapped = im_mask_instances_dst

                    # Depth and IR normalization
                    if (normalize_depth):
                        depth_undist_norm = normalize(im_depth_dst)
                        if (is_8_bit_images):
                            depth_undist_norm = np.true_divide(
                                depth_undist_norm, 256).astype(np.uint8)
                    else:
                        depth_undist_norm = im_depth_dst.copy()
                        depth_undist_norm[depth_undist_norm < low_d] = 0
                        depth_undist_norm[depth_undist_norm > high_d] = 0

                    if (save_ir):
                        if (clip_ir):
                            ir_undist_norm = np.clip(
                                im_ir_dst, 1, 255).astype(np.uint8)
                        elif (log_ir):
                            ir_undist_norm = np.log(im_ir_dst+1) * (
                                255.0 / np.log(65535))
                            ir_undist_norm = ir_undist_norm.astype(np.uint8)
                        else:
                            ir_undist_norm = im_ir_dst.astype(np.uint16)

                    if (zero_replacement):
                        depth_undist_norm[
                            depth_undist_norm == 0] = 255

                    if (pencil_edges):
                        depth_levels = (depth_undist_norm / 16).\
                            astype(np.uint8)
                        depth_pencil = (pencil_filter(depth_levels)/(255*2)) +\
                            0.5
                        depth_undist_norm = (depth_undist_norm *
                                             depth_pencil).astype(np.uint8)

                    if (masks_filling):
                        classmask_remapped, instancemask_remapped = fill_masks(
                            depth_undist_norm,
                            ir_undist_norm,
                            classmask_remapped,
                            instancemask_remapped
                        )

                    depth_undist_orig = depth_undist_norm.copy()
                    if (depth_map_coloring):
                        depth_undist_norm = cv2.applyColorMap(
                            depth_undist_norm,
                            cv2.COLORMAP_JET
                        )

                    classmask_remapped_arr = []
                    instancemask_remapped_arr = []
                    depth_undist_norm_arr = []
                    depth_undist_orig_arr = []
                    ir_undist_norm_arr = []
                    classmask_remapped_path_arr = []
                    instancemask_remapped_path_arr = []
                    classmask_threshold_path_arr = []
                    instancemask_threshold_path_arr = []
                    depth_undist_norm_fname_arr = []
                    ir_undist_norm_fname_arr = []

                    for n in range(scales_size):
                        classmask_remapped_scaled = \
                            apply_resizing_with_borders(
                                classmask_remapped,
                                im_width, im_height, scales[n])
                        instancemask_remapped_scaled = \
                            apply_resizing_with_borders(
                                instancemask_remapped,
                                im_width, im_height, scales[n])
                        depth_undist_norm_scaled = \
                            apply_resizing_with_borders(
                                depth_undist_norm,
                                im_width, im_height, scales[n])
                        depth_undist_orig_scaled = \
                            apply_resizing_with_borders(
                                depth_undist_orig,
                                im_width, im_height, scales[n])
                        if (save_ir):
                            ir_undist_norm_scaled = \
                                apply_resizing_with_borders(
                                    ir_undist_norm,
                                    im_width, im_height, scales[n])

                        for i in range(angles_size):
                            classmask_remapped_arr.append(
                                rotate_image(
                                    classmask_remapped_scaled,
                                    angles[i])
                            )
                            instancemask_remapped_arr.append(
                                rotate_image(
                                    instancemask_remapped_scaled,
                                    angles[i])
                            )
                            depth_undist_norm_arr.append(
                                rotate_image(
                                    depth_undist_norm_scaled,
                                    angles[i])
                            )
                            depth_undist_orig_arr.append(
                                rotate_image(
                                    depth_undist_orig_scaled,
                                    angles[i])
                            )
                            if (save_ir):
                                ir_undist_norm_arr.append(
                                    rotate_image(
                                        ir_undist_norm_scaled,
                                        angles[i])
                                )

                            if (enable_motion_blur):
                                for m in (range(motion_size)):
                                    depth_undist_norm_arr.append(
                                        rotate_image(create_motion_blur(
                                            depth_undist_norm_scaled,
                                            motion[m]), angles[i])
                                    )
                                    if (save_ir):
                                        ir_undist_norm_arr.append(
                                            rotate_image(create_motion_blur(
                                                ir_undist_norm_scaled,
                                                motion[m]), angles[i])
                                        )

                            if (ticam_format):
                                augmentation_postfix = ""
                                sep = ""
                            else:
                                augmentation_postfix = str(angles[i]) + "_" + \
                                    str(scales[n]).replace(".", "_")
                                sep = "_"

                            classmask_remapped_path_arr.append(
                                masks_undist_path + frame_name + sep +
                                augmentation_postfix + '_classes.png')
                            instancemask_remapped_path_arr.append(
                                masks_undist_path + frame_name + sep +
                                augmentation_postfix + '_instances.png')
                            classmask_threshold_path_arr.append(
                                masks_thresh_path + frame_name + sep +
                                augmentation_postfix + '_classes.png')
                            instancemask_threshold_path_arr.append(
                                masks_thresh_path + frame_name + sep +
                                augmentation_postfix + '_instances.png')
                            depth_undist_norm_fname_arr.append(
                                depth_undist_path + frame_name + sep +
                                augmentation_postfix + '.png')
                            if (save_ir):
                                ir_undist_norm_fname_arr.append(
                                    ir_undist_path + frame_name + sep +
                                    augmentation_postfix + '.png')

                            if (enable_motion_blur):
                                for m in (range(motion_size)):
                                    depth_undist_norm_fname_arr.append(
                                        depth_undist_path + frame_name + sep +
                                        augmentation_postfix + "_" +
                                        str(motion[m]) + '_blurred.png')
                                    if (save_ir):
                                        ir_undist_norm_fname_arr.append(
                                            ir_undist_path + frame_name + sep +
                                            augmentation_postfix + "_" +
                                            str(motion[m]) + '_blurred.png')

                        if (depth_map_normals):
                            for i in range(len(depth_undist_norm_arr)):
                                depth_undist_normals = \
                                    generate_surface_normals_new(
                                        depth_undist_norm_arr[i]
                                    )
                                depth_undist_norm_arr[i] = \
                                    depth_undist_normals.copy()

                    im_seq_size = round(
                        len(depth_undist_norm_arr)
                        / angles_size / scales_size)

                    if (ticam_format):
                        num_persons = instancemask_remapped.max()
                    else:
                        pix_persons = classmask_remapped == 1
                        if pix_persons.max():
                            num_persons = instancemask_remapped[pix_persons].max()
                        else:
                            num_persons = 0

                    if(debug_info):
                        print("num_persons: ", num_persons)

                    annotation_recognized = False

                    for im in range(angles_size*scales_size):
                        scales_index = floor(im/angles_size)
                        angles_index = floor(im % angles_size)

                        areas = []
                        bboxes = []
                        segmentations = []
                        for person in range(num_persons):
                            # Preparing and thresholding the original mask
                            person_ind = person + 1
                            filtered_p = np.array(
                                np.logical_and(
                                    classmask_remapped_arr[im] == 1,
                                    instancemask_remapped_arr[im] == person_ind
                                    )).astype(np.uint8)

                            segmentation, bbox, area = \
                                generate_bbox_segmentation(
                                    filtered_p,
                                    is_hull=enable_hull,
                                    area_thresh=area_thresh /
                                    scales[scales_index],
                                    fast_corners=enable_hull)

                            if (
                                (segmentation is None) or
                                (bbox is None) or
                                (area is None)
                            ):
                                bboxes = False
                                break
                            else:
                                segmentations.append(segmentation)
                                bboxes.append(bbox)
                                areas.append(area)

                                annotation_recognized = True

                                if (ticam_format):
                                    info_3d = next(csv_reader_3d)
                                    # print(frame_number-1, info_3d)
                                    input_2d = [frame_number, "Person",
                                                bbox[0], bbox[1],
                                                bbox[0] + bbox[2],
                                                bbox[0] + bbox[3],
                                                "{:.2f}".format(area)]
                                    csv_writer_2d.writerow(input_2d)

                                    input_3d = [frame_number, "Person"]
                                    input_3d.extend(
                                        list(map(float, info_3d[4:13])))
                                    csv_writer_3d.writerow(input_3d)

                                if (generate_statistics):
                                    trunc_thres2 = 0.498
                                    trunc_thres1 = 0.1
                                    occl_thres = 0.1

                                    # Trunc_depth [15]
                                    # Occ_depth [16]
                                    trunc = float(info_3d[15])
                                    occl = float(info_3d[16])

                                    dict_sequence['nannot'] += 1

                                    if (trunc > trunc_thres2):
                                        dict_sequence['ntrunc2'] += 1
                                    elif (trunc_thres1 < trunc < trunc_thres2):
                                        dict_sequence['ntrunc'] += 1
                                    else:
                                        dict_sequence['ntrucnnon'] += 1

                                    if (occl > occl_thres):
                                        dict_sequence['nocc'] += 1

                        if (annotation_recognized):
                            global_ind = scales_index*angles_size+angles_index

                            if (generate_statistics):
                                dict_item = parse_fname(item_name)

                                time = int(dict_item['timestamp_'])
                                if (start_time is None):
                                    start_time = time

                                dict_sequence['nframes'] += 1

                                if (num_persons == 1):
                                    dict_sequence['nframes1'] += 1
                                elif (num_persons == 2):
                                    dict_sequence['nframes2'] += 1
                                elif (num_persons > 2):
                                    dict_sequence['nframesm'] += 1

                                if (num_persons >
                                        dict_sequence['num_persons']):
                                    dict_sequence['num_persons'] = num_persons

                                if not (dict_sequence['add_object']):
                                    dict_sequence['add_object'] = \
                                        classmask_remapped.max() > 1

                            # Saving masks
                            cv2.imwrite(
                                classmask_remapped_path_arr[global_ind],
                                classmask_remapped_arr[global_ind])
                            cv2.imwrite(
                                instancemask_remapped_path_arr[global_ind],
                                instancemask_remapped_arr[global_ind])

                            if (save_masks_thresh):
                                classes_thresh = cv2.threshold(
                                    classmask_remapped_arr[global_ind],
                                    0, 255, cv2.THRESH_BINARY)[1].astype(
                                        np.uint8)
                                instances_thresh = cv2.threshold(
                                    instancemask_remapped_arr[global_ind],
                                    0, 255, cv2.THRESH_BINARY)[1].astype(
                                        np.uint8)
                                cv2.imwrite(
                                    classmask_threshold_path_arr[
                                        global_ind], classes_thresh)
                                cv2.imwrite(
                                    instancemask_threshold_path_arr[
                                        global_ind], instances_thresh)

                            if (save_overlay):
                                overlay_ind = global_ind*im_seq_size
                                if ((ticam_format) or (not normalize_depth)):
                                    overlay_im = normalize(
                                        depth_undist_norm_arr[overlay_ind])
                                    if (is_8_bit_images):
                                        overlay_im = np.true_divide(
                                            overlay_im, 256).astype(np.uint8)
                                else:
                                    overlay_im = \
                                        depth_undist_norm_arr[overlay_ind]
                                if (len(overlay_im.shape) < 3):
                                    overlay_im = cv2.cvtColor(
                                                 overlay_im,
                                                 cv2.COLOR_GRAY2BGR)

                                if enable_hull:
                                    overlay = draw_overlay_image(
                                                overlay_im, bboxes,
                                                segmentations, is_8_bit_images)
                                else:
                                    overlay = draw_overlay_image_new(
                                              overlay_im,
                                              classmask_remapped_arr[
                                                  global_ind],
                                              instancemask_remapped_arr[
                                                  global_ind])

                                if ticam_format:
                                    overlay_name = overlay_path + frame_name + \
                                        "_" + "overlay.png"
                                else:
                                    overlay_name = overlay_path + frame_name + \
                                        "_" + \
                                        str(angles[angles_index]) + \
                                        "_" + \
                                        str(scales[scales_index]).replace(
                                            ".", "_") + "_" + "overlay.png"

                                cv2.imwrite(
                                    overlay_name,
                                    overlay)

                            for _im in range(im_seq_size):
                                im_ind = im*im_seq_size+_im

                                if not (ticam_format):
                                    fname = depth_undist_norm_fname_arr[im_ind]
                                    if (relative_path_annotation):
                                        fname = fname.replace(
                                            depth_undist_path,
                                            "../" + depth_prefix + "/")
                                    dict_annotation = {
                                        "counter_image":
                                        counter_image,
                                        "counter_annotation":
                                        counter_annotation,
                                        "fname": fname,
                                        "width": im_width,
                                        "height": im_height,
                                        "coef_train_val": coef_train_val,
                                        "num_persons": num_persons
                                    }
                                    if not (val_sequences):
                                        is_val_set = \
                                            (randrange(coef_train_val) == 0)

                                    counter_annotation = add_annotation(
                                        json_trainval, json_train, json_val,
                                        segmentations, bboxes, areas,
                                        dict_annotation, is_val_set)

                                # Saving depth and IR
                                fname_index = global_ind*im_seq_size+_im
                                cv2.imwrite(
                                    depth_undist_norm_fname_arr
                                    [fname_index],
                                    depth_undist_norm_arr[fname_index])
                                if (save_ir):
                                    cv2.imwrite(
                                        ir_undist_norm_fname_arr[fname_index],
                                        ir_undist_norm_arr[fname_index])

                                counter_image += 1
                    if (annotation_recognized):
                        seq_count += 1

            print("Annotated " + str(seq_count) +
                  " out of " + str(len_files // (angles_size * scales_size)) +
                  " images.")
            if (seq_count > 0):
                if (ticam_format):
                    csv_reader_3d_file.close()
                    csv_writer_2d_file.close()
                    csv_writer_3d_file.close()

                if (generate_statistics):
                    seq_length = time - start_time
                    nframes = dict_sequence['nframes']

                    dict_sequence['length'] = seq_length
                    csv_writer_stat_sequences.writerow(
                        [
                            dict_sequence['fname'],
                            dict_sequence['nframes'],
                            dict_sequence['nframes1'],
                            dict_sequence['nframes2'],
                            dict_sequence['nframesm'],
                            dict_sequence['nannot'],
                            dict_sequence['ntrunc'],
                            dict_sequence['ntrunc2'],
                        ]
                    )
                    dict_sequence_sum['nframes'] += dict_sequence['nframes']
                    dict_sequence_sum['nframes1'] += dict_sequence['nframes1']
                    dict_sequence_sum['nframes2'] += dict_sequence['nframes2']
                    dict_sequence_sum['nframesm'] += dict_sequence['nframesm']
                    dict_sequence_sum['nannot'] += dict_sequence['nannot']
                    dict_sequence_sum['ntrunc'] += dict_sequence['ntrunc']
                    dict_sequence_sum['ntrunc2'] += dict_sequence['ntrunc2']
                    dict_sequence_sum['nocc'] += dict_sequence['nocc']
                    dict_sequence_sum['length'] += dict_sequence['length']

                    height = int(dict_global_folder['height'])
                    _height = str(height)
                    object_id = dict_global_folder['object_id'].lower()
                    accessory = (dict_global_folder['accessory'].lower() != 'na')

                    if not (height in dict_global):
                        dict_global[height] = {}
                    if not (object_id in dict_global[height]):
                        dict_global[height][object_id] = {
                            'seq': 0,
                            'per_1': 0,
                            'per_1+': 0,
                            'per_2': 0,
                            'per_2+': 0,
                            'per_m': 0,
                            'frames': 0,
                            'length': 0,
                        }

                    if not (_height in dict_summary):
                        dict_summary[_height] = {
                            'seq_total': 0,
                            'per_1': 0,
                            'per_1+': 0,
                            'per_2': 0,
                            'per_2+': 0,
                            'per_m': 0,
                            'frames': 0,
                            'length': 0,
                            'avg_frames': 0,  # frames/seq
                            'avg_length': 0,  # length/seq
                            'fr_': 0,
                            'fr_1': 0,
                            'fr_2': 0,
                            'fr_m': 0,
                            'ann_': 0,
                            'ann_non_tr': 0,
                            'ann_tr_1': 0,
                            'ann_tr_2': 0
                        }

                    dict_global[height][object_id]['seq'] += 1
                    dict_summary[_height]['seq_total'] += 1
                    if (num_persons == 1):
                        if (accessory):
                            dict_global[height][object_id]['per_1+'] += 1
                            dict_summary[_height]['per_1+'] += 1
                        else:
                            dict_global[height][object_id]['per_1'] += 1
                            dict_summary[_height]['per_1'] += 1
                    elif (num_persons == 2):
                        if (accessory):
                            dict_global[height][object_id]['per_2+'] += 1
                            dict_summary[_height]['per_2+'] += 1
                        else:
                            dict_global[height][object_id]['per_2'] += 1
                            dict_summary[_height]['per_2'] += 1
                    elif (num_persons > 2):
                        dict_global[height][object_id]['per_m'] += 1
                        dict_summary[_height]['per_m'] += 1

                    dict_global[height][object_id]['length'] += seq_length
                    dict_global[height][object_id]['frames'] += nframes

                    dict_summary[_height]['length'] += seq_length
                    dict_summary[_height]['frames'] += nframes

                    dict_summary[_height]['fr_'] += dict_sequence['nframes']
                    dict_summary[_height]['fr_1'] += dict_sequence['nframes1']
                    dict_summary[_height]['fr_2'] += dict_sequence['nframes2']
                    dict_summary[_height]['fr_m'] += dict_sequence['nframesm']
                    dict_summary[_height]['ann_'] += dict_sequence['nannot']
                    dict_summary[_height]['ann_non_tr'] += dict_sequence[
                        'ntrucnnon']
                    dict_summary[_height]['ann_tr_1'] += dict_sequence['ntrunc']
                    dict_summary[_height]['ann_tr_2'] += dict_sequence['ntrunc2']
        except KeyboardInterrupt:
            print('Interrupted.')

    if (generate_statistics):
        # Sort the keys to write everything ordered

        dict_global_arranged = {}
        for j in sorted(dict_global):
            _j = str(j)
            dict_global_arranged[j] = {}
            for k in sorted(dict_global[j]):
                csv_writer_stat_persons.writerow(
                    [
                        j, k,
                        dict_global[j][k]['seq'],
                        dict_global[j][k]['per_1'],
                        dict_global[j][k]['per_1+'],
                        dict_global[j][k]['per_2'],
                        dict_global[j][k]['per_2+'],
                        dict_global[j][k]['per_m'],
                        dict_global[j][k]['frames'],
                        dict_global[j][k]['length'],
                    ]
                )
                dict_global_sum['seq'] += dict_global[j][k]['seq']
                dict_global_sum['per_1'] += dict_global[j][k]['per_1']
                dict_global_sum['per_1+'] += dict_global[j][k]['per_1+']
                dict_global_sum['per_2'] += dict_global[j][k]['per_2']
                dict_global_sum['per_2+'] += dict_global[j][k]['per_2+']
                dict_global_sum['per_m'] += dict_global[j][k]['per_m']
                dict_global_sum['frames'] += dict_global[j][k]['frames']
                dict_global_sum['length'] += dict_global[j][k]['length']

            dict_summary[_j]['diff_pers'] = len(dict_global[j])
            dict_summary[_j]['avg_frames'] = "{:.2f}".format(
                (float(dict_summary[_j]['frames']) /
                 float(dict_summary[_j]['seq_total']))).replace(".", ",")
            dict_summary[_j]['avg_length'] = "{:.2f}".format(
                (float(dict_summary[_j]['length']) /
                 float(dict_summary[_j]['seq_total']))).replace(".", ",")

            dict_summary['sum']['seq_total'] += dict_summary[_j]['seq_total']
            dict_summary['sum']['per_1'] += dict_summary[_j]['per_1']
            dict_summary['sum']['per_1+'] += dict_summary[_j]['per_1+']
            dict_summary['sum']['per_2'] += dict_summary[_j]['per_2']
            dict_summary['sum']['per_2+'] += dict_summary[_j]['per_2+']
            dict_summary['sum']['per_m'] += dict_summary[_j]['per_m']

            dict_summary['sum']['frames'] += dict_summary[_j]['frames']
            dict_summary['sum']['length'] += dict_summary[_j]['length']
            dict_summary['sum']['fr_'] += dict_summary[_j]['fr_']
            dict_summary['sum']['fr_1'] += dict_summary[_j]['fr_1']
            dict_summary['sum']['fr_2'] += dict_summary[_j]['fr_2']
            dict_summary['sum']['fr_m'] += dict_summary[_j]['fr_m']
            dict_summary['sum']['ann_'] += dict_summary[_j]['ann_']
            dict_summary['sum']['ann_non_tr'] += dict_summary[_j]['ann_non_tr']
            dict_summary['sum']['ann_tr_1'] += dict_summary[_j]['ann_tr_1']
            dict_summary['sum']['ann_tr_2'] += dict_summary[_j]['ann_tr_2']

        dict_summary['sum']['avg_frames'] = "{:.2f}".format(
            (float(dict_summary['sum']['frames']) /
             float(dict_summary['sum']['seq_total']))).replace(".", ",")
        dict_summary['sum']['avg_length'] = "{:.2f}".format(
            (float(dict_summary['sum']['length']) /
             float(dict_summary['sum']['seq_total']))).replace(".", ",")

        dict_summary_rows = [
            [
                "Height",
                "Total number of Sequences",
                "With 1 Person",
                "With 1 Person + object",
                "With 2 Persons",
                "With 2 Persons + object",
                "With >2 Persons",
                "Number of different Persons",
                "",
                "Total number of Frames",
                "Average sequence length [frames]",
                "Average sequence length [msec]",
                "",
                "Total number of annotated Frames",
                "Frames with 1 person",
                "Frames with 2 persons",
                "Frames with more persons",
                "",
                "Total number of annotated person frames",
                "Non truncted 3D boxes",
                "3D boxes with (trunc < 0.50)",
                "3D boxes with (trunc >= 0.50)"
            ]
        ]

        for j in sorted(dict_summary):
            dict_summary_rows.append([
                j,
                dict_summary[j]['seq_total'],
                dict_summary[j]['per_1'],
                dict_summary[j]['per_1+'],
                dict_summary[j]['per_2'],
                dict_summary[j]['per_2+'],
                dict_summary[j]['per_m'],
                dict_summary[j]['diff_pers'] if (j != 'sum') else "",
                "",
                dict_summary[j]['frames'],
                dict_summary[j]['avg_frames'],
                dict_summary[j]['avg_length'],
                "",
                dict_summary[j]['fr_'],
                dict_summary[j]['fr_1'],
                dict_summary[j]['fr_2'],
                dict_summary[j]['fr_m'],
                "",
                dict_summary[j]['ann_'],
                dict_summary[j]['ann_non_tr'],
                dict_summary[j]['ann_tr_1'],
                dict_summary[j]['ann_tr_2'],
            ])

        # Transpose
        dict_summary_rows_save = list(map(list, zip(*dict_summary_rows)))

        csv_writer_stat_persons.writerow(
            [
                "", "",
                dict_global_sum['seq'],
                dict_global_sum['per_1'],
                dict_global_sum['per_1+'],
                dict_global_sum['per_2'],
                dict_global_sum['per_2+'],
                dict_global_sum['per_m'],
                dict_global_sum['frames'],
                dict_global_sum['length'],
            ]
        )

        csv_writer_stat_sequences.writerow(
            [
                dict_sequence_sum['fname'],
                dict_sequence_sum['nframes'],
                dict_sequence_sum['nframes1'],
                dict_sequence_sum['nframes2'],
                dict_sequence_sum['nframesm'],
                dict_sequence_sum['nannot'],
                dict_sequence_sum['ntrunc'],
                dict_sequence_sum['ntrunc2'],
            ]
        )

        for j in dict_summary_rows_save:
            csv_writer_stat_summary.writerow(j)

        csv_writer_stat_summary_file.close()
        csv_writer_stat_sequences_file.close()
        csv_writer_stat_persons_file.close()

        # Parsing of dict_global

    if not (ticam_format):
        if (shuffle_images):
            shuffle(json_train['images'])
            shuffle(json_val['images'])

            shuffle(json_train['annotations'])
            shuffle(json_val['annotations'])

        save_json(annotations_path,
                  json_trainval_fname, json_train_fname, json_val_fname,
                  json_trainval, json_train, json_val)

        print("Making TAR.")
        tar_name = os.path.basename(os.path.dirname(save_path)) + '.tar'
        try:
            tar = tarfile.open(save_path + tar_name,
                               mode='w', format=tarfile.GNU_FORMAT)
            tar.add(annotations_path, arcname='annotations')
            tar.add(depth_undist_path, arcname='depth')
            if (save_ir):
                tar.add(ir_undist_path, arcname='ir')
            tar.close()
        except Exception:
            print("Something went wrong.")
        else:
            print("Saved as", save_path + tar_name)


if __name__ == "__main__":
    main()
