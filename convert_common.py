import cv2
import json
from math import ceil
import numpy as np
import shutil
from shapely.geometry import Polygon, MultiPolygon
import struct
from datetime import datetime
import zlib


def get_timestamp():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def parse_dir(name):
    name_splitted = name.split("_")
    try:
        return {
            "object": name_splitted[0],
            "object_id": name_splitted[1],
            "accessory": name_splitted[2],
            "action": name_splitted[3],
            "direction": name_splitted[4],
            "ground_truth": name_splitted[5],
            "height": name_splitted[7],
            "timestamp": name_splitted[8],
            "camera_id": name_splitted[9],
            "calib_id": name_splitted[10],
        }
    except Exception:
        return None


def parse_fname(name):
    name_splitted = name.split("_")
    try:
        return {
            "object": name_splitted[0],
            "object_id": name_splitted[1],
            "accessory": name_splitted[2],
            "action": name_splitted[3],
            "direction": name_splitted[4],
            "ground_truth": name_splitted[5],
            "height": name_splitted[7],
            "timestamp": name_splitted[8],
            "camera_id": name_splitted[9],
            "calib_id": name_splitted[10],
            "timestamp_": name_splitted[11],
            "raw_frame_id": name_splitted[12],
            "frame_id": name_splitted[13],
        }
    except Exception:
        return None


def normalize(im_src, make_disparity=False):
    # Azure range: WFOV 2x2 binned 	512x512 120°x120° 0.25 - 2.88 m 12.8 ms
    # Through away the values outside the range

    low_d = 250
    high_d = 2880

    im_src[im_src < 250] = 0
    im_src[im_src > 2880] = 0

    im_depth = (im_src.astype(np.int16) - (low_d + 1))
    im_depth[im_depth < 0] = 0
    im_depth_norm = im_depth / (high_d - low_d)

    if (make_disparity):
        disp_scaling = 65535
        im_disp = im_depth.astype(np.float32)
        im_disp[im_disp > 0] = (
            (1-disp_scaling)/(high_d - low_d - 1)) * (
            im_disp[im_disp > 0]-1)+disp_scaling
        im_dst = np.uint16(im_disp)
    else:
        im_dst = np.uint16(im_depth_norm * 65535.0)

    return im_dst


def generate_point_cloud(im_depth, camera_matrix, scaling=1):
    height, width = im_depth.shape
    xx, yy = np.meshgrid(np.array(range(width))+1, np.array(range(height))+1)
    # color camera parameters
    cc = camera_matrix[0:2, 2] * scaling
    fc = np.diag(camera_matrix[0:2, 0:2]) * scaling
    x3 = np.multiply((xx - cc[0]), im_depth) / fc[0]
    y3 = np.multiply((yy - cc[1]), im_depth) / fc[1]

    x3 = x3.astype(np.float)
    y3 = y3.astype(np.float)
    z3 = im_depth.astype(np.float)

    points = np.array([x3, y3, z3]).transpose(1, 2, 0)
    return points


def rotate_point_cloud(points, rotation_matrix):
    _points = []
    for p in points:
        _p = np.dot(rotation_matrix[0], p)
        _points.append(_p)
    _points = np.array(_points).reshape(-1, 3)
    return _points


def write_point_cloud(filename, xyz_points, rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3, 'Input XYZ points should be Nx3 float'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape, 'Input RGB colors should be \
        Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename, 'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n' % xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",
                                        xyz_points[i, 0],
                                        xyz_points[i, 1],
                                        xyz_points[i, 2],
                                        rgb_points[i, 0].tostring(),
                                        rgb_points[i, 1].tostring(),
                                        rgb_points[i, 2].tostring())))
    fid.close()


def generate_transformed_image(image, fx, fy, cx, cy, scaling, R, t):
    im_size = image.shape
    image_new = np.zeros(im_size)

    rows, cols = np.nonzero(image)
    for y, x in zip(rows, cols):
        point = np.array([x-cx, y-cy, image[y][x]])
        point_new = point.dot(R)
        point_new += t

        x = point_new[0] * fx/point_new[2]
        y = point_new[1] * fy/point_new[2]
        z = point_new[2] * scaling

        x = int(round(x + cx))
        y = int(round(y + cy))
        if ((y > 0) and (y <= im_size[0] - 1) and
           (x > 0) and (x <= im_size[1] - 1)):
            image_new[y][x] = int(z)

    image_new = image_new.astype(np.uint8)
    image_new = cv2.medianBlur(image_new, 5)
    return image_new


def create_motion_blur(image, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel /= kernel_size
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def pencil_filter(im_src):
    morph_size = 1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2 * morph_size + 1,
                                         2 * morph_size + 1),
                                        (morph_size,
                                         morph_size))
    im_dst = cv2.erode(im_src, element)
    for y in range(len(im_dst)):
        for x in range(len(im_dst[y])):
            val_src = im_src[y][x]
            val_dst = im_dst[y][x]
            if (val_dst == val_src == 0):
                im_dst[y][x] = 255
            else:
                im_dst[y][x] = (int(val_dst) * 255) / int(val_src)

    im_dst = im_dst - 255
    # cv2.threshold(im_dst, 1, 255, cv2.THRESH_BINARY, im_dst)

    return im_dst.astype(np.uint8)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1],
                            flags=cv2.INTER_NEAREST)
    return result


def item_name_vizta_convention(file_name):
    item_name = file_name.split('.png')[0]

    orig_len = 14
    item_name_parameters = item_name.split("_")
    mod = orig_len - len(item_name_parameters)
    item_name_parameters[-1 + mod] = str(int(item_name_parameters[-1 + mod]))
    item_name_parameters[-2 + mod] = str(int(item_name_parameters[-2 + mod]))
    item_name_parameters[-3 + mod] = str(int(item_name_parameters[-3 + mod]))
    item_name_parameters[-1 + mod] = item_name_parameters[-1 + mod].zfill(5)
    item_name_parameters[-2 + mod] = item_name_parameters[-2 + mod].zfill(6)
    item_name_parameters[-3 + mod] = item_name_parameters[-3 + mod].zfill(7)
    item_name_new = "_".join(item_name_parameters)
    return item_name_new


def get_frame_number(file_name):
    num = file_name.split('_')[-1]
    if (num.isnumeric()):
        return int(num)
    else:
        raise ValueError


def get_crc(src):
    return 'P' + str(zlib.crc32(str(src).encode()))[-6:]


def folder_name_dataset_conversion(folder_name):
    num = folder_name.split('_')
    seq_id = '-'.join([num[4], num[5]])

    new_fname = [num[3], seq_id, get_crc(num[8]), num[8], num[7]]
    return '_'.join(new_fname)


def frame_name_dataset_conversion(file_name):
    num = file_name.split('_')
    seq_id = '-'.join([num[4], num[5]])
    new_fname = [num[3], seq_id, get_crc(num[8]), num[8], num[7], num[10], num[13]]
    return '_'.join(new_fname)


def traverse_ranges(r, row_depth, row_ir,
                    row_classes, row_instances,
                    maxm_thr, fill_thr, right=True):
    if (right):
        m = +1
    else:
        m = -1

    # Check all pixels till the first big difference
    st = (r[1] if right else r[0])  # Beginning of range -- to the left
    val = row_depth[st]
    val_ir = row_ir[st]
    len_row = len(row_depth)
    cycle = True
    _st = st
    while (cycle):
        for n in range(maxm_thr):
            _st = _st + (n * m)
            valid = ((_st < len_row) if right else (st >= 0))
            if (valid):
                _val = row_depth[_st]
                _val_ir = row_ir[_st]
                if ((val - fill_thr) <= _val <= (val + fill_thr)):
                    row_classes[_st] = row_classes[st]
                    row_instances[_st] = row_instances[st]
                    val = _val
                    val_ir = _val_ir
                else:
                    cycle = False
            else:
                cycle = False
        cycle = False

    return row_classes, row_instances


def generate_ranges(row_mask):
    # If we have valid pixels in mask
    ranges = []
    if (row_mask.max() > 0):
        _r = list(np.where(row_mask > 0))[0]
        len_r = len(_r)
        if (len_r == 1):
            ranges.append([_r[0], _r[0]])
        elif (len_r == 2):
            ranges.append([_r[0], _r[1]])
        else:
            start_pix = _r[0]
            end_pix = _r[0]
            for _l in range(len_r)[1:]:
                if (_l < len_r - 1):
                    if (_r[_l] == end_pix + 1):
                        end_pix = _r[_l]
                    else:
                        ranges.append([start_pix, end_pix])
                        start_pix = _r[_l]
                        end_pix = _r[_l]
                else:
                    end_pix = _r[_l]
                    ranges.append([start_pix, end_pix])
    return ranges


def generate_pre_bbox(filtered_p, bbox_thr):
    w, h = filtered_p.shape
    _valid = np.argwhere(filtered_p > 0).transpose()
    _min_x = np.min(_valid[1])
    _min_y = np.min(_valid[0])
    _max_x = np.max(_valid[1])
    _max_y = np.max(_valid[0])

    _bbox = (
        (_min_y - bbox_thr) if (_min_y >= bbox_thr) else 0,
        (_max_y + bbox_thr) if (_max_y <= h-bbox_thr) else h-1,
        (_min_x - bbox_thr) if (_min_x >= bbox_thr) else 0,
        (_max_x + bbox_thr) if (_max_x <= w-bbox_thr) else w-1,
    )

    return _bbox


def generate_filtered_p(ind_class, ind_instance, classes, instances):
    return np.array(
        np.logical_and(
            classes == ind_class,
            instances == ind_instance
            )).astype(np.uint8)


def fill_masks(depth, ir, classes, instances):
    bbox_thr = 3  # Threshold for bounding box "expansion"
    fill_thr = 3  # Difference between reference and neighbor values
    maxm_thr = 3  # Maximum movement in one direction -- protection from spikes

    num_persons = instances.max()
    _depth = depth
    _ir = cv2.medianBlur(ir, 5)

    for person in range(num_persons):
        # Preparing and thresholding the original mask
        person_ind = person + 1

        filtered_p = generate_filtered_p(1, person_ind, classes, instances)
        if (filtered_p.max() > 0):
            _bbox = generate_pre_bbox(filtered_p, bbox_thr)

            # Subset images, on which the search is going to be done
            subset_depth = _depth[_bbox[0]:_bbox[1], _bbox[2]:_bbox[3]]
            subset_ir = _ir[_bbox[0]:_bbox[1], _bbox[2]:_bbox[3]]
            subset_mask = filtered_p[_bbox[0]:_bbox[1], _bbox[2]:_bbox[3]]
            subset_classes = classes[_bbox[0]:_bbox[1], _bbox[2]:_bbox[3]]
            subset_instances = instances[_bbox[0]:_bbox[1], _bbox[2]:_bbox[3]]

            h_subset, w_subset = subset_depth.shape

            # For every row
            for y in range(h_subset):
                row_im = subset_depth[y]
                row_ir = subset_ir[y]
                row_mask = subset_mask[y]

                ranges = generate_ranges(row_mask)

                for r in ranges:
                    ret_classes, ret_instances = \
                        traverse_ranges(r,
                                        row_im, row_ir,
                                        subset_classes[y],
                                        subset_instances[y],
                                        maxm_thr, fill_thr,
                                        False)

                    ret_classes, ret_instances = \
                        traverse_ranges(r,
                                        row_im, row_ir,
                                        ret_classes,
                                        ret_instances,
                                        maxm_thr, fill_thr,
                                        True)

                    subset_classes[y] = ret_classes
                    subset_instances[y] = ret_instances

            # Transposing as the easier solution to adjust colums

            subset_depth = subset_depth.transpose()
            subset_ir = subset_ir.transpose()
            subset_mask = subset_mask.transpose()
            subset_classes = subset_classes.transpose()
            subset_instances = subset_instances.transpose()

            h_subset, w_subset = subset_depth.shape

            # For every "row"
            for y in range(h_subset):
                row_im = subset_depth[y]
                row_ir = subset_ir[y]
                row_mask = subset_mask[y]

                ranges = generate_ranges(row_mask)

                for r in ranges:
                    ret_classes, ret_instances = \
                        traverse_ranges(r,
                                        row_im, row_ir,
                                        subset_classes[y],
                                        subset_instances[y],
                                        maxm_thr, fill_thr,
                                        False)

                    ret_classes, ret_instances = \
                        traverse_ranges(r,
                                        row_im, row_ir,
                                        ret_classes,
                                        ret_instances,
                                        maxm_thr, fill_thr,
                                        True)

                    subset_classes[y] = ret_classes
                    subset_instances[y] = ret_instances

            subset_depth = subset_depth.transpose()
            subset_ir = subset_ir.transpose()
            subset_mask = subset_mask.transpose()
            subset_classes = subset_classes.transpose()
            subset_instances = subset_instances.transpose()

    return [classes, instances]


'''
Input: thresholded image with the object for segmentation
'''


def generate_bbox_segmentation(src,
                               is_hull=False,
                               area_thresh=5000,
                               fast_corners=False):
    segmentation = None
    bbox = None
    area = None

    if (fast_corners):
        corners_approx_flag = cv2.CHAIN_APPROX_SIMPLE
    else:
        corners_approx_flag = cv2.CHAIN_APPROX_NONE

    # Find contours from thresholded image
    contours, hierarchy = cv2.findContours(
        src,
        cv2.RETR_CCOMP,
        corners_approx_flag
        )

    # Transform all contours in polygons, if they exist
    polygons = []
    if (contours):
        for contour in contours:
            if (len(contour) > 2):
                contour = np.squeeze(contour)
                poly = Polygon(contour)
                if (fast_corners):
                    polygons.append(poly)
                else:
                    if not is_hull:
                        poly = poly.simplify(1.0, preserve_topology=False)
                    if (poly.geom_type == 'MultiPolygon'):
                        for _poly in poly.geoms:
                            polygons.append(_poly)
                    elif (poly.geom_type == "Polygon"):
                        polygons.append(poly)

        if (polygons):
            multi_poly = MultiPolygon(polygons)

            if (multi_poly.area > area_thresh):
                segmentation = []

                # Getting segmentation and bounding box based on hull flag
                if (is_hull):
                    hull = multi_poly.convex_hull
                    segmentation = [
                        np.array(hull.exterior.coords).ravel().tolist()
                        ]
                    x, y, max_x, max_y = hull.bounds
                    area = hull.area
                else:
                    for poly in polygons:
                        segm = np.array(poly.exterior.coords).ravel().tolist()
                        if (len(segm) > 0):
                            segmentation.append(segm)
                    x, y, max_x, max_y = multi_poly.bounds
                    area = multi_poly.area

                # Bounding box transformation for the training-accepted format
                if (x < 0.0):
                    x = 0.0
                if (y < 0.0):
                    y = 0.0
                bbox_width = max_x - x
                bbox_height = max_y - y
                bbox = (x, y, bbox_width, bbox_height)

    return segmentation, bbox, area


def draw_overlay_image(im, bboxes, segmentations,
                       is_8_bit_images, color=(0, 255, 0)):
    overlay = None
    if (im.ndim == 2):
        overlay = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    else:
        overlay = im.copy()

    if not (is_8_bit_images):
        color *= 256

    color_bbox = color
    thickness_bbox = 2
    color_segment = color
    thickness_segment = 1

    if (bboxes):
        for i in range(len(bboxes)):
            x1_bboxes = int(bboxes[i][0])
            y1_bboxes = int(bboxes[i][1])
            x2_bboxes = x1_bboxes + int(bboxes[i][2])
            y2_bboxes = y1_bboxes + int(bboxes[i][3])
            start_point_bbox = (x1_bboxes, y1_bboxes)
            end_point_bbox = (x2_bboxes, y2_bboxes)
            overlay = cv2.rectangle(overlay,
                                    start_point_bbox, end_point_bbox,
                                    color_bbox, thickness_bbox)
        for k in range(len(segmentations)):
            for i in range(len(segmentations[k])):
                range_segments = int(len(segmentations[k][i])/2 - 1)
                for n in range(range_segments):
                    x1_segment = int(segmentations[k][i][n*2])
                    y1_segment = int(segmentations[k][i][n*2+1])
                    x2_segment = int(segmentations[k][i][(n+1)*2])
                    y2_segment = int(segmentations[k][i][(n+1)*2+1])
                    start_point_bbox = (x1_segment, y1_segment)
                    end_point_bbox = (x2_segment, y2_segment)
                    cv2.line(overlay,
                             start_point_bbox,
                             end_point_bbox,
                             color_segment,
                             thickness_segment)
    return overlay


colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255),
          (255, 255, 0), (255, 0, 255), (0, 255, 255))


def draw_overlay_image_new(im, classes, instances):

    overlay = None
    if (im.ndim == 2):
        overlay = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    else:
        overlay = im.copy()

    im_masks = np.zeros(overlay.shape, dtype=np.uint8)
    classes_max = classes.max()
    instances_max = instances.max()

    for cl in range(classes_max):
        for inst in range(instances_max):
            filtered_p = generate_filtered_p(cl+1, inst+1,
                                             classes, instances)
            im_masks[filtered_p > 0] = colors[cl]

    overlay = cv2.addWeighted(im_masks, 0.5, overlay, 1.0, gamma=0)
    return overlay


def add_annotation(json_trainval, json_train, json_val,
                   segmentations, bboxes, areas,
                   dict_annotation, is_val_set):
    image_id = dict_annotation["counter_image"]
    json_images = {
        "id": image_id,
        "width": dict_annotation["width"],
        "height": dict_annotation["height"],
        "file_name": dict_annotation["fname"],
    }
    is_crowd = 0  # Should remain 0
    json_trainval['images'].append(json_images)
    if (is_val_set):
        json_val['images'].append(json_images)
    else:
        json_train['images'].append(json_images)

    if (bboxes is not False):
        for person in range(dict_annotation["num_persons"]):
            annotation_id = dict_annotation["counter_annotation"]
            json_annotations = {
                'segmentation': segmentations[person],
                'iscrowd': is_crowd,
                'image_id': image_id,
                'category_id': 1,
                'id': annotation_id,
                'bbox': bboxes[person],
                'area': areas[person]
            }
            json_trainval['annotations'].append(json_annotations)
            if (is_val_set):
                json_val['annotations'].append(json_annotations)
            else:
                json_train['annotations'].append(json_annotations)
            dict_annotation["counter_annotation"] += 1

    return dict_annotation["counter_annotation"]


def save_json(json_annotations_path,
              json_trainval_fname, json_train_fname, json_val_fname,
              json_trainval, json_train, json_val):
    print("Done with annotating.")
    print("Saving json_trainval_fname.")
    with open(json_annotations_path + json_trainval_fname, 'w') as outfile:
        json.dump(json_trainval, outfile, indent=4)
    print("Saving json_train_fname.")
    with open(json_annotations_path + json_train_fname, 'w') as outfile:
        json.dump(json_train, outfile, indent=4)
    print("Saving json_val_fname.")
    with open(json_annotations_path + json_val_fname, 'w') as outfile:
        json.dump(json_val, outfile, indent=4)


def remove_intermediate_folders(depth_path, ir_path, masks_path):
    shutil.rmtree(depth_path)
    shutil.rmtree(ir_path)
    shutil.rmtree(masks_path)


def add_borders(image, im_height, im_width, scale):
    width_div = ceil(im_width/scale)
    height_div = ceil(im_height/scale)

    if (scale > 1):
        if (width_div % 2 != 0):
            width_div += 1
        if (height_div % 2 != 0):
            height_div += 1

    return cv2.copyMakeBorder(
        image,

        left=ceil((im_width - width_div)/2),
        right=ceil((im_width - width_div)/2),
        top=ceil((im_height - height_div)/2),
        bottom=ceil((im_height - height_div)/2),

        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )


def apply_resizing_with_borders(im, width, height, scale):
    width_scale = round(width/scale)
    height_scale = round(height/scale)

    # if (width_scale % 2 != 0): width_scale += 1
    # if (height_scale % 2 != 0): height_scale += 1

    im_scaled = cv2.resize(
        im,
        dsize=(width_scale, height_scale),
        interpolation=cv2.INTER_NEAREST
    )
    im_scaled = add_borders(
        im_scaled, width, height, scale)
    return im_scaled


def replace_slash(text):
    text = text.replace("\\", "/")
    text = text.replace("//", "/")
    return text


def generate_surface_normals(im_src):
    kernel_x = np.array(
        [
            [0.5, 0.0, -0.5]
        ],
        np.float32)

    kernel_y = np.array(
        [
            [0.5],
            [0.0],
            [-0.5]
        ],
        np.float32)

    im_blurred = cv2.GaussianBlur(im_src, (5, 5), 0)
    im_dst_x = cv2.filter2D(im_blurred, -1, kernel_x).astype(float)
    im_dst_y = cv2.filter2D(im_blurred, -1, kernel_y).astype(float)

    return im_dst_x, im_dst_y


def generate_surface_normals_new(im_src):
    zy, zx = np.gradient(im_src)
    # You may also consider using Sobel
    # to get a joint Gaussian smoothing
    # and differentation to reduce noise
    # zx = cv2.Sobel(im_src, cv2.CV_64F, 1, 0, ksize=5)
    # zy = cv2.Sobel(im_src, cv2.CV_64F, 0, 1, ksize=5)

    normals = np.dstack((-zx, -zy, np.ones_like(im_src)))
    n = np.linalg.norm(normals, axis=2)
    normals[:, :, 0] /= n
    normals[:, :, 1] /= n
    normals[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normals += 1
    normals /= 2
    normals *= 255

    return normals


def generate_height_map(im_src):
    subset = np.where(im_src > 0)
    minval = np.min(im_src[subset])
    maxval = np.max(im_src[subset])
    im_dst = np.zeros(im_src.shape)
    im_dst[subset] = np.abs(
                     (maxval - (im_src[subset] - minval + 1) - minval + 1))
    minval = np.min(im_dst[subset])
    maxval = np.max(im_dst[subset])

    return im_dst.astype(np.uint8)
