import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.resnet import ResNet
from utils.box_utils import match, crop, make_anchors
from modules.swin_transformer import SwinTransformer
import pdb


class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = torch.tensor([0.25,1], device='cuda')
        self.gamma = gamma
        self.reduction = 'mean'

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction = self.reduction
        )


class PredictionModule(nn.Module):
    def __init__(self, cfg, coef_dim=32):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.coef_dim = coef_dim

        self.upfeature = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                       nn.ReLU(inplace=True))
        self.bbox_layer = nn.Conv2d(256, len(cfg.aspect_ratios) * 4, kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(256, len(cfg.aspect_ratios) * self.num_classes, kernel_size=3, padding=1)
        self.coef_layer = nn.Sequential(nn.Conv2d(256, len(cfg.aspect_ratios) * self.coef_dim,
                                                  kernel_size=3, padding=1),
                                        nn.Tanh())

    def forward(self, x):
        x = self.upfeature(x)
        conf = self.conf_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)
        box = self.bbox_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
        coef = self.coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        return conf, box, coef


class ProtoNet(nn.Module):
    def __init__(self, coef_dim):
        super().__init__()
        self.proto1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.proto2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, coef_dim, kernel_size=1, stride=1),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.proto1(x)
        x = self.upsample(x)
        x = self.proto2(x)
        return x


class FPN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.lat_layers = nn.ModuleList([nn.Conv2d(x, 256, kernel_size=1) for x in self.in_channels])
        self.pred_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                                        nn.ReLU(inplace=True)) for _ in self.in_channels])

        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                                                              nn.ReLU(inplace=True)),
                                                nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                                                              nn.ReLU(inplace=True))])

        self.upsample_module = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)])

    def forward(self, backbone_outs):
        p5_1 = self.lat_layers[2](backbone_outs[2])
        p5_upsample = self.upsample_module[1](p5_1)

        p4_1 = self.lat_layers[1](backbone_outs[1]) + p5_upsample
        p4_upsample = self.upsample_module[0](p4_1)

        p3_1 = self.lat_layers[0](backbone_outs[0]) + p4_upsample

        p5 = self.pred_layers[2](p5_1)
        p4 = self.pred_layers[1](p4_1)
        p3 = self.pred_layers[0](p3_1)

        p6 = self.downsample_layers[0](p5)
        p7 = self.downsample_layers[1](p6)

        return p3, p4, p5, p6, p7




class Yolact(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.coef_dim = 32

        if cfg.__class__.__name__ == 'iee101_dataset':
            self.backbone = ResNet(layers=(3, 4, 23, 3))
            self.fpn = FPN(in_channels=(512, 1024, 2048))
        elif cfg.__class__.__name__ == 'iee_swin_dataset':
            self.backbone = SwinTransformer()
            self.fpn = FPN(in_channels=(192, 384, 768))

        self.proto_net = ProtoNet(coef_dim=self.coef_dim)
        self.prediction_layers = PredictionModule(cfg, coef_dim=self.coef_dim)

        self.anchors = []
        fpn_fm_shape = [math.ceil(cfg.img_size / stride) for stride in (8, 16, 32, 64, 128)]
        for i, size in enumerate(fpn_fm_shape):
            self.anchors += make_anchors(self.cfg, size, size, self.cfg.scales[i])

        if cfg.mode == 'train':
            self.semantic_seg_conv = nn.Conv2d(256, cfg.num_classes - 1, kernel_size=1)

    def load_weights(self, weight, cuda):
        if cuda:
            state_dict = torch.load(weight)
        else:
            state_dict = torch.load(weight, map_location='cpu')

        for key in list(state_dict.keys()):
            if self.cfg.mode != 'train' and key.startswith('semantic_seg_conv'):
                del state_dict[key]

        self.load_state_dict(state_dict, strict=True)
        print(f'Model loaded with {weight}.\n')
        print(f'Number of all parameters: {sum([p.numel() for p in self.parameters()])}\n')

    def forward(self, img, box_classes=None, masks_gt=None):
        outs = self.backbone(img)
        outs = self.fpn(outs[1:4])
        proto_out = self.proto_net(outs[0])  # feature map P3
        proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

        class_pred, box_pred, coef_pred = [], [], []

        for aa in outs:
            class_p, box_p, coef_p = self.prediction_layers(aa)
            class_pred.append(class_p)
            box_pred.append(box_p)
            coef_pred.append(coef_p)

        #print('IMAGE : ',img,'Shape : ',img.size())
        #print('Box_classes : ',box_classes)
        #print('Masks_gt : ',masks_gt)
        class_pred = torch.cat(class_pred, dim=1)
        box_pred = torch.cat(box_pred, dim=1)
        coef_pred = torch.cat(coef_pred, dim=1)
        #print('CLASS_PRED : ',class_pred,'Shape : ',class_pred.size())
        #print('BOX_PRED : ',box_pred,'Shape : ',box_pred.size())
        #print('COEF_PRED : ',coef_pred,'Shape : ',coef_pred.size())

        if self.training:
            seg_pred = self.semantic_seg_conv(outs[0])
            return self.compute_loss(class_pred, box_pred, coef_pred, proto_out, seg_pred, box_classes, masks_gt)
        else:
            class_pred = F.softmax(class_pred, -1)
            return class_pred, box_pred, coef_pred, proto_out

    def compute_loss(self, class_p, box_p, coef_p, proto_p, seg_p, box_class, mask_gt):
        device = class_p.device
        class_gt = [None] * len(box_class)
        batch_size = box_p.size(0)

        if isinstance(self.anchors, list):
            self.anchors = torch.tensor(self.anchors, device=device).reshape(-1, 4)

        num_anchors = self.anchors.shape[0]

        all_offsets = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)
        conf_gt = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)
        anchor_max_gt = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)
        anchor_max_i = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)

        for i in range(batch_size):
            box_gt = box_class[i][:, :-1]
            class_gt[i] = box_class[i][:, -1].long()

            all_offsets[i], conf_gt[i], anchor_max_gt[i], anchor_max_i[i] = match(self.cfg, box_gt,
                                                                                  self.anchors, class_gt[i])
        #print('all_offsets : ',all_offsets,'Shape : ',all_offsets.size())
        #print('conf_gt : ',conf_gt,'Shape : ',conf_gt.size())
        #print('anchor_max_gt : ',anchor_max_gt,'Shape : ',anchor_max_gt.size())
        #print('anchor_max_i : ',anchor_max_i,'Shape : ',anchor_max_i.size())
        #print('class_gt : ',class_gt)
        #print('box_gt : ',box_gt)
        # all_offsets: the transformed box coordinate offsets of each pair of anchor and gt box
        # conf_gt: the foreground and background labels according to the 'pos_thre' and 'neg_thre',
        #          '0' means background, '>0' means foreground.
        # anchor_max_gt: the corresponding max IoU gt box for each anchor
        # anchor_max_i: the index of the corresponding max IoU gt box for each anchor
        assert (not all_offsets.requires_grad) and (not conf_gt.requires_grad) and \
               (not anchor_max_i.requires_grad), 'Incorrect computation graph, check the grad.'

        # only compute losses from positive samples
        pos_bool = conf_gt > 0
        #misclass_bool = conf_gt == 0 
        #misclass_bool = class_p[1] >= class_p[0]
        #misclass_bool = torch.transpose(misclass_bool,0,1)
        #misclass_bool[pos_bool] = 0
        #misclass_bool[conf_gt < 0] = 0
        #final_bool = pos_bool + misclass_bool
        #print('final_bool : ',final_bool,'Shape : ',final_bool.size())
        #print('Misclass bool : ',misclass_bool,'Shape : ',misclass_bool.size())
        loss_c = self.category_loss(class_p, conf_gt, pos_bool)
        #print('Loss_c :',loss_c)
        loss_b = self.box_loss(box_p, all_offsets, pos_bool)
        #print('LOSS_B : ',loss_b)
        loss_m = self.lincomb_mask_loss(pos_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_gt)
        #loss_m = loss_b
        #print('LOSS_M : ',loss_m)
        loss_s = self.semantic_seg_loss(seg_p, mask_gt, class_gt)
        #print('LOSS_S : ',loss_s)
        loss_n = self.surf_n_loss(box_p, all_offsets, pos_bool) # Repeats box loss by now
        #print('LOSS_N : ',loss_n)
        return loss_c, loss_b, loss_m, loss_s, loss_n


    def category_loss(self, class_p, conf_gt, pos_bool, np_ratio=5):
        # Compute max conf across batch for hard negative mining
        batch_conf = class_p.reshape(-1, self.cfg.num_classes)
        #conf_gt_rs = conf_gt.reshape(-1)
        #conf_gt_rs = conf_gt_rs.type(torch.LongTensor)
        #print('Class_p : ',class_p,'Shape : ',class_p.size())
        #print('conf_gt : ',conf_gt,'Shape : ',conf_gt.size())
        #print('pos_bool : ',pos_bool,'Shape : ',pos_bool.size())
        #print('Batch_conf : ',batch_conf,'Shape : ',batch_conf.size())
        #print('Conf_gt_rs : ',conf_gt,'Shape : ',conf_gt.size())
        batch_conf_max = batch_conf.max()
        #print('batch_conf_max : ',batch_conf_max,'Shape : ',batch_conf_max.size())
        mark = torch.log(torch.sum(torch.exp(batch_conf - batch_conf_max), 1)) + batch_conf_max - batch_conf[:, 0]
        #print('Mark : ',mark,'Shape : ',mark.size())
        # Hard Negative Mining
        mark = mark.reshape(class_p.size(0), -1)
        mark[pos_bool] = 0  # filter out pos boxes
        mark[conf_gt < 0] = 0  # filter out neutrals (conf_gt = -1)
        #print('Mark after changes : ',mark,'Shape : ',mark.size())
        _, idx = mark.sort(1, descending=True)
        _, idx_rank = idx.sort(1)
        #print('IDX : ',idx,'Shape : ',idx.size())
        #print('idx_rank : ',idx_rank,'Shape : ',idx_rank.size())

        num_pos = pos_bool.long().sum(1, keepdim=True)
        num_neg = torch.clamp(np_ratio * num_pos, max=pos_bool.size(1) - 1)
        neg_bool = idx_rank < num_neg.expand_as(idx_rank)
        #print('num_pos : ',num_pos,'Shape : ',num_pos.size())
        #print('num_neg : ',num_neg,'Shape : ',num_neg.size())
        #print('neg_bool : ',neg_bool,'Shape : ',neg_bool.size())

        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg_bool[pos_bool] = 0
        neg_bool[conf_gt < 0] = 0  # Filter out neutrals
        #print('Neg bool after changes : ',neg_bool,'Shape : ',neg_bool.size())

        negative_bool = conf_gt == 0
        #print('Negative bool :',negative_bool,'Shape : ',negative_bool.size())
        # Confidence Loss Including Positive and Negative Examples
        class_p_mined = class_p[(pos_bool + negative_bool)].reshape(-1, self.cfg.num_classes)
        class_gt_mined = conf_gt[(pos_bool + negative_bool)]

        #if(num_pos.sum() != 0):
        #   div = num_pos.sum()
        #else:
        #   div = 1
        #print('Shape:class_gt : ',class_gt_mined.size())
        #print('Class_p_mined : ',class_p_mined,'Shape : ',class_p_mined.size())
        #print('Class_gt_mined : ',class_gt_mined,'Shape : ',class_gt_mined.size())
        #print('Num pos :', num_pos.sum())
        #print(self.cfg.conf_alpha * F.cross_entropy(class_p_mined, class_gt_mined, reduction='sum'))
        #print(focal_loss)
        focal_loss = FocalLoss()
        loss_class = focal_loss.forward(class_p_mined , class_gt_mined)
        print(loss_class)
        #loss_class = loss_class.sum()
        #num_of_pred = class_gt_mined.size(0)
        #loss_class = loss_class / num_of_pred
        #return loss_class
        #return self.cfg.conf_alpha * F.cross_entropy(class_p_mined, class_gt_mined, reduction='sum') / num_pos.sum()

    def box_loss(self, box_p, all_offsets, pos_bool):
        num_pos = pos_bool.sum()
        pos_box_p = box_p[pos_bool, :]
        pos_offsets = all_offsets[pos_bool, :]
        #return self.cfg.bbox_alpha * F.smooth_l1_loss(box_p , all_offsets , reduction = 'mean')
        return self.cfg.bbox_alpha * F.smooth_l1_loss(pos_box_p, pos_offsets, reduction='mean')

    def lincomb_mask_loss(self, pos_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_gt):
        proto_h, proto_w = proto_p.shape[1:3]
        total_pos_num = pos_bool.sum()
        loss_m = 0
        #print('Proto_p : ',proto_p,'Shape : ',proto_p.size())
        #print('Coef_p : ',coef_p,'Shape : ',coef_p.size())
        #print('anchor_max_i : ',anchor_max_i,'Shape : ',anchor_max_i.size())
        #print('mask_gt : ',mask_gt)
        #print('anchor_max_gt : ',anchor_max_gt)
        for i in range(coef_p.size(0)):
            # downsample the gt mask to the size of 'proto_p'
            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            #print('downsampled_masks : ',downsampled_masks,'Shape : ',downsampled_masks.size())
            downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()
            # binarize the gt mask because of the downsample operation
            #print('downsampled_masks 1 : ',downsampled_masks,'Shape : ',downsampled_masks.size())
            downsampled_masks = downsampled_masks.gt(0.5).float()
            #print('downsampled_masks 2 : ',downsampled_masks,'Shape : ',downsampled_masks.size())

            pos_anchor_i = anchor_max_i[i][pos_bool[i]]
            #print('pos_anchor_i : ',pos_anchor_i,'Shape : ',pos_anchor_i.size())
            pos_anchor_box = anchor_max_gt[i][pos_bool[i]]
            #print('pos_anchor_box : ',pos_anchor_box,'Shape : ',pos_anchor_box.size())
            pos_coef = coef_p[i][pos_bool[i]]
            #print('pos_coef : ',pos_coef,'Shape : ',pos_coef.size())

            if pos_anchor_i.size(0) == 0:
                continue

            # If exceeds the number of masks for training, select a random subset
            old_num_pos = pos_coef.size(0)
            #print('old_num_pos : ',old_num_pos)
            if old_num_pos > self.cfg.masks_to_train:
                perm = torch.randperm(pos_coef.size(0))
                #print('prem : ',perm,'Shape : ',perm.size())
                select = perm[:self.cfg.masks_to_train]
                #print('select : ',select,'Shape : ',select.size())
                pos_coef = pos_coef[select]
                #print('pos_coef : ',pos_coef,'Shape : ',pos_coef.size())
                pos_anchor_i = pos_anchor_i[select]
                #print('pos_anchor_i : ',pos_anchor_i,'Shape : ',pos_anchor_i.size())
                pos_anchor_box = pos_anchor_box[select]
                #print('pos_anchor_box : ',pos_anchor_box,'Shape : ',pos_anchor_box.size())

            num_pos = pos_coef.size(0)
            #print('num_pos : ',num_pos)
            pos_mask_gt = downsampled_masks[:, :, pos_anchor_i]
            #print('pos_mask_gt : ',pos_mask_gt,'Shape : ',pos_mask_gt.size())
            # mask assembly by linear combination
            # @ means dot product
            mask_p = torch.sigmoid(proto_p[i] @ pos_coef.t())
            #print('mask_p : ',mask_p,'Shape : ',mask_p.size())
            mask_p = crop(mask_p, pos_anchor_box)  # pos_anchor_box.shape: (num_pos, 4)
            #print('mask_p 1 : ',mask_p,'Shape : ',mask_p.size())
            # TODO: grad out of gt box is 0, should it be modified?
            # TODO: need an upsample before computing loss?
            mask_loss = F.binary_cross_entropy(torch.clamp(mask_p, 0, 1), pos_mask_gt, reduction='none')
            # mask_loss = -pos_mask_gt*torch.log(mask_p) - (1-pos_mask_gt) * torch.log(1-mask_p)
            #print('Mask_loss : ',mask_loss,'Shape : ',mask_loss.size())
            # Normalize the mask loss to emulate roi pooling's effect on loss.
            anchor_area = (pos_anchor_box[:, 2] - pos_anchor_box[:, 0]) * (pos_anchor_box[:, 3] - pos_anchor_box[:, 1])
            mask_loss = mask_loss.sum(dim=(0, 1)) / anchor_area
            #print('anchor area : ',anchor_area,'Shape : ',anchor_area.size())
            #print('mask_loss 1 : ',mask_loss,'Shape : ',mask_loss.size())
            if old_num_pos > num_pos:
                mask_loss *= old_num_pos / num_pos
            #print('mask loss 2 : ',mask_loss,'Shape : ',mask_loss.size())
            loss_m += torch.sum(mask_loss)
            #print('loss_m : ',loss_m,'Shape : ',loss_m.size())
        return self.cfg.mask_alpha * loss_m / proto_h / proto_w / total_pos_num

    def semantic_seg_loss(self, segmentation_p, mask_gt, class_gt):
        # Note classes here exclude the background class, so num_classes = cfg.num_classes - 1
        batch_size, num_classes, mask_h, mask_w = segmentation_p.size()
        loss_s = 0

        for i in range(batch_size):
            cur_segment = segmentation_p[i]
            cur_class_gt = class_gt[i]

            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (mask_h, mask_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.gt(0.5).float()

            # Construct Semantic Segmentation
            segment_gt = torch.zeros_like(cur_segment, requires_grad=False)
            for j in range(downsampled_masks.size(0)):
                segment_gt[cur_class_gt[j]] = torch.max(segment_gt[cur_class_gt[j]], downsampled_masks[j])

            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_gt, reduction='sum')

        return self.cfg.semantic_alpha * loss_s / mask_h / mask_w / batch_size

    def surf_n_loss(self, box_p, all_offsets, pos_bool):
        num_pos = pos_bool.sum()
        pos_box_p = box_p[pos_bool, :]
        pos_offsets = all_offsets[pos_bool, :]
        #return 0.001 * F.smooth_l1_loss(box_p , all_offsets , reduction='mean')
        return 0.001 * F.smooth_l1_loss(pos_box_p, pos_offsets, reduction='sum') / num_pos
