import torch
import torch.nn.functional as F
from torch import Tensor, cat, index_select,\
    sigmoid, exp, meshgrid, linspace, stack, log, min, max, matmul, abs, prod, where, argmax
import numpy as np

from config import NOobject_COEFF, \
    COORD_COEFF, \
    IGNORE_THRESH, \
    ANCHORS, \
    epsilon

epsilon = 0.001
np.argmax(3, 3)


def loss(predictions, targets, targets_length, image_size: int, average=True):
    """Calculate the loss function given the predictions, the targets, the length of each target and the image size.
    Args:
        predictions: (Tensor) the raw prediction tensor. Size is [B, N_PRED, NUM_ATTRIB],
                where B is the batch size;
                N_PRED is the total number of predictions, equivalent to 3*N_GRID*N_GRID for each scale;
                NUM_ATTRIB is the number of attributes, determined in config.py.
                coordinates is in format cxcywh and is local (raw).
                objectectness is in logit.
                class score is in logit.
        targets:   (Tensor) the tensor of ground truths (targets). Size is [B, N_target_max, NUM_ATTRIB].
                where N_target_max is the max number of targets in this batch.
                If a certain sample has targets fewer than N_target_max, zeros are filled at the tail.
        targets_length: (Tensor) a 1D tensor showing the number of the targets for each sample. Size is [B, ].
        image_size: (int) the size of the training image.
        average: (bool) the flag of whether the loss is summed loss or average loss over the batch size.
    Return:
        the total loss
        """

    # generate the no-objectectness mask. mask_noobject has size of [B, N_PRED]
    mask_noobject = noobject_mask_fn(predictions, targets)
    target_t_1d, index_predicted_object = build_targets(targets, targets_length, image_size)
    mask_noobject = noobject_mask_filter(mask_noobject, index_predicted_object)

    # calculate the no-objectectness loss
    predicted_confidence_logit = predictions[..., 4]
    target_zero = torch.zeros(predicted_confidence_logit.size(), device=predicted_confidence_logit.device)
    # target_noobject = target_zero + (1 - mask_noobject) * 0.5
    predicted_confidence_logit = predicted_confidence_logit - (1 - mask_noobject) * 1e7
    noobject_loss = F.binary_cross_entropy_with_logits(predicted_confidence_logit, target_zero, reduction='sum')

    # select the predictions corresponding to the targets
    n_batch, n_pred, _ = predictions.size()
    preds_1d = predictions.view(n_batch * n_pred, -1)
    preds_object = preds_1d.index_select(0, index_predicted_object)

    # calculate the coordinate loss
    coord_loss = F.mse_loss(preds_object[..., :4], target_t_1d[..., :4], reduction='sum')
    # assert not torch.isnan(coord_loss)

    # calculate the objectectness loss
    predicted_confidence_object_logit = preds_object[..., 4]
    target_one = torch.ones(predicted_confidence_object_logit.size(), device=predicted_confidence_object_logit.device)
    object_loss = F.binary_cross_entropy_with_logits(predicted_confidence_object_logit, target_one, reduction='sum')

    # calculate the classification loss
    classification_loss = F.binary_cross_entropy_with_logits(preds_object[..., 5:], target_t_1d[..., 5:],
                                                             reduction='sum')

    # total loss
    total_loss = noobject_loss * NOobject_COEFF + object_loss + classification_loss + coord_loss * COORD_COEFF

    if average:
        total_loss = total_loss / n_batch

    return total_loss, coord_loss, object_loss, noobject_loss, classification_loss


def noobject_mask_fn(predictions, target):
    """pred is a 3D tensor with shape
    (num_batch, NUM_ANCHORS_PER_SCALE*num_grid*num_grid, NUM_ATTRIB). The raw data has been converted.
    target is a 3D tensor with shape
    (num_batch, max_num_objectect, NUM_ATTRIB).
     The max_num_objectects depend on the sample which has max num_objectects in this minibatch"""
    num_batch, num_pred, num_attrib = predictions.size()
    ious = iou_batch(predictions[..., :4], target[..., :4])  # in cxcywh format
    # for each pred bbox, find the target box which overlaps with it (without zero centered) most, and the iou value.
    max_ious, max_ious_index = max(ious, dim=2)
    noobject_indicator = torch.where((max_ious - IGNORE_THRESH) > 0, torch.zeros_like(max_ious),
                                     torch.ones_like(max_ious))
    return noobject_indicator


def noobject_mask_filter(mask_noobject: Tensor, index_object_1d: Tensor):
    n_batch, n_pred = mask_noobject.size()
    mask_noobject = mask_noobject.view(-1)
    filter_ = torch.zeros(mask_noobject.size(), device=mask_noobject.device)
    mask_noobject.scatter_(0, index_object_1d, filter_)
    mask_noobject = mask_noobject.view(n_batch, -1)
    return mask_noobject


anchors = Tensor([[1, 2], [3, 4], [5, 6]])


def build_targets(ground_trues_boxes, target_length, grid_size, image_size):
    """get the index of the predictions corresponding to the targets;
    and put targets from different sample into one dimension (flatten), getting rid of the tails;
    and convert coordinates to local.
    Args:
        ground_trues_boxes: (tensor) the tensor of ground truths boxes. Size is [B, N_target_max, NUM_ATTRIB].
                    where B is the batch size;
                    N_target_max is the max number of targets in this batch;
                    NUM_ATTRIB is the number of attributes, determined in config.py.
                    coordinates is in format cxcywh and is global.
                    If a certain sample has targets fewer than N_target_max, zeros are filled at the tail.
        target_length: (Tensor) a 1D tensor showing the number of the targets for each sample. Size is [B, ].
        image_size: (int) the size of the training image.
    :return
        target_t_flat: (tensor) the flattened and local target. Size is [N_target_total, NUM_ATTRIB],
                            where N_target_total is the total number of targets in this batch.
        index_object_1d: (tensor) the tensor of the indices of the predictions corresponding to the targets.
                            The size is [N_target_total, ]. Note the indices have been added the batch number,
                            therefore when the predictions are flattened, the indices can directly find the prediction.
    """
    # find the anchor box which has max IOU (zero centered) with the targets
    area1, area2 = prod(anchors, -1), prod(ground_trues_boxes[..., 2:], -1)
    intersection_area = prod(min(anchors, ground_trues_boxes[..., :2].unsqueeze(-2)), -1)
    anchors_suitability_measured_by_iou = intersection_area / (area1 + area2 - intersection_area + epsilon)
    index_anchor = argmax(anchors_suitability_measured_by_iou, dim=-2)

    # find the corresponding prediction's index for the anchor box with the max IOU
    scale = index_anchor // 3
    index_anchor_by_scale = index_anchor - scale * 3
    grid_offsets = ground_trues_boxes[..., :2] // grid_size
    large_scale_mask = (scale <= 1).long()
    med_scale_mask = (scale <= 0).long()

    # calculate t_x and t_y
    target_centers = map(lambda c: log(c / (1. - c)),
            (ground_trues_boxes[..., :2] / grid_size - grid_offsets).clamp(epsilon, 1 - epsilon))

    # calculate t_w and t_h
    corresponding_anchors = index_select(anchors, 0, index_anchor.view(-1)).reshape_as(index_anchor)
    target_sizes = log((ground_trues_boxes[..., 2:] / corresponding_anchors).clamp(min=epsilon))

    # the raw target tensor
    return cat((target_centers, target_sizes), dim=-1)


def flat_target(target):
    index_object = \
        large_scale_mask * (image_size // strides_selection[2]) ** 2 * 3 + \
        med_scale_mask * (image_size // strides_selection[1]) ** 2 * 3 + \
        n_grid ** 2 * index_anchor_by_scale + n_grid * grid_y + grid_x

    # aggregate processed targets and the corresponding prediction index from different batches in to one dimension
    n_batch = ground_trues_boxes.size(0)
    n_pred = sum([(image_size // s) ** 2 for s in strides_selection]) * 3

    index_object_1d = []
    target_t_flat = []

    for i_batch in range(n_batch):
        v = index_object[i_batch]
        t = target[i_batch]
        l = target_length[i_batch]
        index_object_1d.append(v[:l] + i_batch * n_pred)
        target_t_flat.append(t[:l])

    index_object_1d = cat(index_object_1d)
    target_t_flat = cat(target_t_flat)

    return target_t_flat, index_object_1d


def count_iou(boxes1, boxes2):
    area1, area2 = prod(boxes1[..., 2:], -1), prod(boxes2[..., 2:], -1)
    boxes2.unsqueeze_(-2)
    intersection_area = \
        prod((min(boxes1[..., :2] - boxes1[..., 2:] / 2, boxes2[..., :2] - boxes2[..., 2:] / 2) -
              max(boxes1[..., :2] + boxes1[..., 2:] / 2, boxes2[..., :2] + boxes2[..., 2:] / 2)).clamp(min=0), -1)
    boxes2.squeeze_(-2)
    return intersection_area / (area1 + area2 - intersection_area + epsilon)
