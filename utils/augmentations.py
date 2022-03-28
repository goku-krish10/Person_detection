import cv2
import numpy as np
import random

from config import norm_mean, norm_std
import pdb


def random_mirror(img, masks, boxes):
    if random.randint(0, 1):
        _, width = img.shape
        img = img[:, ::-1]
        masks = masks[:, :, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]

    return img, masks, boxes


def clip_box(hw, boxes):
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], a_min=0, a_max=hw[1] - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], a_min=0, a_max=hw[0] - 1)
    return boxes


def remove_small_box(boxes, masks, labels, area_limit):
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = box_areas > area_limit
    return boxes[keep], masks[keep], labels[keep]


def to_01_box(hw, boxes):
    boxes[:, [0, 2]] /= hw[1]
    boxes[:, [1, 3]] /= hw[0]
    return boxes


def crop(ori_h, crop_h, ori_w, crop_w, img, masks, boxes, labels, keep_ratio=0.3):
    num_boxes = boxes.shape[0]
    box_x1, box_y1, box_x2, box_y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    box_areas = (box_x2 - box_x1) * (box_y2 - box_y1)

    ii = 0
    cut_out = True
    while cut_out:
        ii += 1
        if ii > 1000:
            return None, None, None, None

        random_x1 = random.randint(0, ori_w - crop_w)
        random_y1 = random.randint(0, ori_h - crop_h)

        new_x1 = np.tile(random_x1, (num_boxes, 1)).astype('float32')
        new_y1 = np.tile(random_y1, (num_boxes, 1)).astype('float32')
        new_x2 = np.tile(random_x1 + crop_w, (num_boxes, 1)).astype('float32')
        new_y2 = np.tile(random_y1 + crop_h, (num_boxes, 1)).astype('float32')

        min_x1 = np.max(np.concatenate([new_x1, box_x1.reshape(num_boxes, -1)], axis=1), axis=1)
        min_y1 = np.max(np.concatenate([new_y1, box_y1.reshape(num_boxes, -1)], axis=1), axis=1)
        max_x2 = np.min(np.concatenate([new_x2, box_x2.reshape(num_boxes, -1)], axis=1), axis=1)
        max_y2 = np.min(np.concatenate([new_y2, box_y2.reshape(num_boxes, -1)], axis=1), axis=1)

        inter_w = np.clip((max_x2 - min_x1), a_min=0, a_max=10000)
        inter_h = np.clip((max_y2 - min_y1), a_min=0, a_max=10000)

        inter_area = inter_w * inter_h
        keep = (inter_area / box_areas) > keep_ratio

        if keep.any():
            box_part = np.stack([min_x1, min_y1, max_x2, max_y2, labels], axis=0).T
            boxes_remained = box_part[keep]
            masks_remained = masks[keep]

            boxes_remained[:, [0, 2]] -= random_x1
            boxes_remained[:, [1, 3]] -= random_y1

            img_cropped = img[int(new_y1[0]): int(new_y2[0]), int(new_x1[0]): int(new_x2[0]), :]
            masks_remained = masks_remained[:, int(new_y1[0]): int(new_y2[0]), int(new_x1[0]): int(new_x2[0])]

            cut_out = False

    return img_cropped, masks_remained, boxes_remained[:, :4], boxes_remained[:, 4]


def random_crop(img, masks, boxes, labels, crop_ratio):
    if random.randint(0, 1):
        return img, masks, boxes, labels
    else:
        ori_h, ori_w, _ = img.shape
        crop_h = int(random.uniform(crop_ratio[0], crop_ratio[1]) * ori_h)
        crop_w = int(random.uniform(crop_ratio[0], crop_ratio[1]) * ori_w)

        return crop(ori_h, crop_h, ori_w, crop_w, img, masks, boxes, labels)


def pad_to_square(img, masks=None, boxes=None, during_training=False):
    img_h, img_w = img.shape[:2]
    if img_h == img_w:
        return (img, masks, boxes) if during_training else img
    else:
        pad_size = max(img_h, img_w)
        pad_img = np.zeros((pad_size, pad_size, 3), dtype='float32')
        pad_img[:, :, :] = norm_mean

        if during_training:
            pad_masks = np.zeros((masks.shape[0], pad_size, pad_size), dtype='float32')

            if img_h < img_w:
                random_y1 = random.randint(0, img_w - img_h)
                pad_img[random_y1: random_y1 + img_h, :, :] = img
                pad_masks[:, random_y1: random_y1 + img_h, :] = masks
                boxes[:, [1, 3]] += random_y1

            if img_h > img_w:
                random_x1 = random.randint(0, img_h - img_w)
                pad_img[:, random_x1: random_x1 + img_w, :] = img
                pad_masks[:, :, random_x1: random_x1 + img_w] = masks
                boxes[:, [0, 2]] += random_x1

            return pad_img, pad_masks, boxes
        else:
            pad_img[0: img_h, 0: img_w, :] = img
            return pad_img


def multi_scale_resize(img, masks=None, boxes=None, resize_range=None, during_training=False):
    assert img.shape[0] == img.shape[1], 'Error, image is not square in <multi_scale_resize>'

    if during_training:
        ori_size = img.shape[0]
        resize_size = random.randint(resize_range[0], resize_range[1]) * 32
        img = cv2.resize(img, (resize_size, resize_size))
        scale = resize_size / ori_size
        boxes *= scale

        masks = masks.transpose((1, 2, 0))
        masks = cv2.resize(masks, (resize_size, resize_size))

        # OpenCV resizes a (w,h,1) array to (s,s), so fix it
        if len(masks.shape) == 2:
            masks = np.expand_dims(masks, 0)
        else:
            masks = masks.transpose((2, 0, 1))

        return img, masks, boxes
    else:
        return cv2.resize(img, (resize_range, resize_range))


def to_train_size(img, masks, boxes, labels, train_size):
    img_size = img.shape[0]

    if img_size == train_size:
        return img, masks, boxes, labels
    elif img_size < train_size:
        pad_img = np.zeros((train_size, train_size, 3), dtype='float32')
        pad_masks = np.zeros((masks.shape[0], train_size, train_size), dtype='float32')
        pad_img[:, :, :] = norm_mean
        random_y1 = random.randint(0, train_size - img_size)
        random_x1 = random.randint(0, train_size - img_size)
        pad_img[random_y1: random_y1 + img_size, random_x1: random_x1 + img_size, :] = img
        pad_masks[:, random_y1: random_y1 + img_size, random_x1: random_x1 + img_size] = masks
        boxes[:, [1, 3]] += random_y1
        boxes[:, [0, 2]] += random_x1
        return pad_img, pad_masks, boxes, labels
    else:
        return crop(img_size, train_size, img_size, train_size, img, masks, boxes, labels)


"""def generate_surface_normals(img, step=1):
    #np.save("orig_img.npy", img);
    _img = cv2.medianBlur(img, ksize=3)
    #np.save("median_img.npy", _img)
    grad_x = cv2.Sobel(_img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(_img, cv2.CV_32F, 0, 1, ksize=3)
    normals = np.dstack((-grad_x, -grad_y, np.ones_like(_img)))
    n = np.linalg.norm(normals, axis=2)
    _n = np.full(img.shape, False)
    _n[::step, ::step] = True
    cond = (n != 1.0) & _n
    normals[:, :, 0] = np.where(cond, (normals[:, :, 0]/n + 1.0)/2.0, 0.0)
    normals[:, :, 1] = np.where(cond, (normals[:, :, 1]/n + 1.0)/2.0, 0.0)
    normals[:, :, 2] = np.where(cond, (normals[:, :, 2]/n + 1.0)/2.0, 0.0)

    if (False):
        _normals = normals * 255;
        med_img = _img / 255;

        cv2.imwrite("normals.exr", normals)
        cv2.imwrite("normals.png", _normals)

        cv2.imwrite("median.exr", _img)
        cv2.imwrite("median.png", med_img)

    if (False):
        np.save("orig_img.npy", img);
        np.save("median_img.npy", _img);
        np.save("grad_x_img.npy", grad_x);
        np.save("grad_y_img.npy", grad_y);
        np.save("normals_img.npy", normals);

    if (False):
        cv2.imwrite("orig_img.exr", img);
        cv2.imwrite("median_img.exr", _img);
        cv2.imwrite("grad_x_img.exr", grad_x);
        cv2.imwrite("grad_y_img.exr", grad_y);
        cv2.imwrite("normals_img.exr", normals);

    normals = np.transpose(normals, (2, 0, 1))
    return normals"""

def normalize_and_toRGB(img):
    img_orig = img
    #cv2.imwrite('Orig_img.png',img_orig)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #img = cv2.GaussianBlur(img,(11,11),0)
    #print(img.dtype)
    #cv2.imwrite('Color_cvt_img.png',img)
    #img = cv2.merge((img,img,img))
    #img =  cv2.applyColorMap(img, cv2.COLORMAP_JET)
    #cv2.imwrite('Norm_img.png',img)
    img = (img - norm_mean) / norm_std
    img = img[:, :, (2, 1, 0)]
    #cv2.imwrite('Norm_img.png',img)
    img = np.transpose(img, (2, 0, 1))
    return img

def val_aug(img, val_size):
    img = img.astype('float32')
    img = pad_to_square(img, during_training=False)
    img = normalize_and_toRGB(img)
    return img


def train_aug(img, masks, boxes, labels, train_size):
    img = img.astype('float32')

    #img, masks, boxes = random_mirror(img, masks, boxes)
    #print('After RANDOM MIRROR')
    #print('IMG_ : ',img)
    #print('BOXES_ : ',boxes)
    #print('MASKS_ : ',masks)
    #print('LABLES_ : ',labels)
    if img is None:
        return None, None, None, None
    img, masks, boxes = pad_to_square(img, masks, boxes, during_training=True)
    #print('AFTER PAD TO SQUARE')
    #print('IMG_ : ',img)
    #print('BOXES_ : ',boxes)
    #print('MASKS_ : ',masks)
    #print('LABLES_ : ',labels)
    img, masks, boxes, labels = to_train_size(img, masks, boxes, labels, train_size)
    #print('AFTER TO_TRAIN_SIZE')
    #print('IMG_ : ',img)
    #print('BOXES_ : ',boxes)
    #print('MASKS_ : ',masks)
    #print('LABLES_ : ',labels)
    if img is None:
        return None, None, None, None
    boxes = clip_box(img.shape[:2], boxes)
    #print('AFTER CLIP BOX')
    #print('IMG_ : ',img)
    #print('BOXES_ : ',boxes)
    #print('MASKS_ : ',masks)
    #print('LABLES_ : ',labels)
    #boxes, masks, labels = remove_small_box(boxes, masks, labels, area_limit=20)
    #print('AFTER REMOVE SMALL BOXES')
    #print('IMG_ : ',img)
    #print('BOXES_ : ',boxes)
    #print('MASKS_ : ',masks)
    #print('LABLES_ : ',labels)
    if boxes.shape[0] == 0:
        return None, None, None, None
    assert boxes.shape[0] == masks.shape[0] == labels.shape[0], 'Error, unequal boxes, masks or labels number.'
    boxes = to_01_box(img.shape[:2], boxes)
    img = normalize_and_toRGB(img)

    return img, masks, boxes, labels


def show_ann(img, masks, boxes, labels):
    masks_semantic = masks * (labels[:, None, None] + 1)  # expand class_ids' shape for broadcasting
    # The color of the overlap area is different because of the '%' operation.
    masks_semantic = masks_semantic.astype('int').sum(axis=0) % 80
    from config import COLORS
    color_masks = COLORS[masks_semantic].astype('uint8')
    img_u8 = img.astype('uint8')
    img_fused = cv2.addWeighted(color_masks, 0.4, img_u8, 0.6, gamma=0)

    for i in range(boxes.shape[0]):
        cv2.rectangle(img_fused, (int(boxes[i, 0]), int(boxes[i, 1])),
                      (int(boxes[i, 2]), int(boxes[i, 3])), (0, 255, 0), 1)

    print(f'\nimg shape: {img.shape}')
    print('----------------boxes----------------')
    print(boxes)
    print('----------------labels---------------')
    print(labels, '\n')
    cv2.imshow('aa', img_fused)
    cv2.waitKey()
