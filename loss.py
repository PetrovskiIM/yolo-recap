import torch
import torch.nn.functional as F
from torch import Tensor, cat, index_select,zeros, \
    sigmoid, exp, meshgrid, linspace, stack, log, min, max, matmul, abs, prod, where, argmax, ones_like, zeros_like
import numpy as np

# from config import NOobject_COEFF, \
#     COORD_COEFF, \
#     IGNORE_THRESH, \
#     ANCHORS, \
#     epsilon

epsilon = 0.001
np.argmax(3, 3)
ignore_thresh = 0.5


def loss(predictions, targets, image_size: int, average=True):
    # generate the no-objectectness mask. mask_noobject has size of [B, N_PRED]
    predictions_1, predictions_2, predictions_3 = predictions
    """
    we have predited_boxes and ground trues boxes
    predicted box -> target box that overlaps with predicted box with iou > ignore thresh
    
    at training time we only want one bounding box predictor to be responsible for each object.
    
    
    """
    mask_noobject = noobject_mask_fn(predictions, targets) # batch size x predictions size x targets size

    target_t_1d, index_predicted_object = build_targets(targets, image_size)
    mask_noobject = noobject_mask_filter(mask_noobject, index_predicted_object)

    # calculate the no-objectectness loss
    predicted_confidence_logit = predictions[..., 4]
    target_zero = zeros(predicted_confidence_logit.size())
    predicted_confidence_logit = predicted_confidence_logit - (1 - mask_noobject) * 1e7
    noobject_loss = F.binary_cross_entropy_with_logits(predicted_confidence_logit, target_zero, reduction='sum')

    # select the predictions corresponding to the targets
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
    """loss function only penalizes classification error if and object is present in that grid cell.
    It also only penalizes bounding box coordinate error if that predictor is responsiable for the ground 
    trues box(i.e. has the highest iou of any predictor in that grid cell"""
    # total loss
    # total_loss = noobject_loss * NOobject_COEFF + object_loss + classification_loss + coord_loss * COORD_COEFF

    # if average:
    #     total_loss = total_loss / n_batch
    #
    # return total_loss, coord_loss, object_loss, noobject_loss, classification_loss


def noobject_mask_fn(predictions, target):
    max_ious, max_ious_index = max(count_iou(predictions[..., :4], target[..., :4]), dim=-1)
    return ((max_ious - ignore_thresh) > 0).float()


def noobject_mask_filter(mask_noobject, index_object_1d):
    n_batch, n_pred = mask_noobject.size()
    mask_noobject = mask_noobject.view(-1)
    filter_ = torch.zeros(mask_noobject.size(), device=mask_noobject.device)
    mask_noobject.scatter_(0, index_object_1d, filter_)
    mask_noobject = mask_noobject.view(n_batch, -1)
    return mask_noobject


anchors = Tensor([[1, 2], [3, 4], [5, 6]])


def build_targets(ground_trues_boxes, grid_sizes, head_indexes, anchors):
    def inverse_of_sigmoid(tensor):
        return log(tensor / (1. - tensor))
    area1, area2 = prod(ground_trues_boxes[..., 2:], -1), prod(anchors, -1)
    intersection_area = prod(min(ground_trues_boxes[..., 2:].unsqueeze(-2), anchors), -1)
    anchors_suitability_measured_by_iou = intersection_area / (
            area1.unsqueeze(-1) + area2 - intersection_area + epsilon)
    # anchors_correspondence = argmax(anchors_suitability_measured_by_iou, dim=-1) # batch size x number of boxes
    anchors_assigment = (anchors_suitability_measured_by_iou == max(anchors_suitability_measured_by_iou))
    anchors_index_assigment = argmax(anchors_suitability_measured_by_iou, dim=-1)

    output = []
    for grid_size in grid_sizes:
        cells_offsets = stack(meshgrid(linspace(0, 1 - 1 / grid_size[0], grid_size[0]),
                                       linspace(0, 1 - 1 / grid_size[1], grid_size[1]).t()), -1)
        cells_coordinates = cells_offsets // (1. / Tensor(grid_size))
        ground_boxes_coordinates = ground_trues_boxes[..., :2].unsqueeze(-2).unsqueeze(-2) // (1. / Tensor(grid_size))
        coordinates_match = (ground_boxes_coordinates == cells_coordinates)
        corresponding_anchors = index_select(anchors, 0, anchors_index_assigment.view(-1)).view(
            list(anchors_index_assigment.size())+[2])
        target_sizes = log((ground_trues_boxes[..., 2:] / corresponding_anchors).clamp(min=epsilon))
        presense_of_objects = coordinates_match[..., 0] & coordinates_match[..., 1]
        object_mask = anchors_assigment.view(-1).repeat(grid_size).view(
            list(presense_of_objects.size()) + [len(anchors)]) & presense_of_objects.unsqueeze(-1)
        if max(torch.sum(object_mask, dim=1)) > 1:
            raise ValueError
        grid_offsets = ground_trues_boxes[..., :2] // (1 / Tensor(grid_size))
        sigmoid_of_target_centers = (ground_trues_boxes[..., :2] / Tensor(grid_size) - grid_offsets).clamp(epsilon, 1 - epsilon)
        target_centers = inverse_of_sigmoid(sigmoid_of_target_centers)

        output.append(cat((matmul(target_centers.unsqueeze(-1)[..., 0], object_mask.float()),
                           matmul(target_centers.unsqueeze(-1)[..., 1], object_mask.float()),
                           matmul(target_sizes.unsqueeze(-1)[..., 0], object_mask.float()),
                           matmul(target_sizes.unsqueeze(-1)[..., 0], object_mask.float())), -1))
    return output


def count_iou(boxes1, boxes2):
    area1, area2 = prod(boxes1[..., 2:], -1), prod(boxes2[..., 2:], -1)
    boxes2.unsqueeze_(-2)
    intersection_area = \
        prod((min(boxes1[..., :2] - boxes1[..., 2:] / 2, boxes2[..., :2] - boxes2[..., 2:] / 2) -
              max(boxes1[..., :2] + boxes1[..., 2:] / 2, boxes2[..., :2] + boxes2[..., 2:] / 2)).clamp(min=0), -1)
    boxes2.squeeze_(-2)
    return intersection_area / (area1 + area2 - intersection_area + epsilon)
