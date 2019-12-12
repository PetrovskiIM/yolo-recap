import torch
from torch import Tensor, cat, sigmoid, exp, arange, meshgrid, linspace, stack
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.nn import ModuleList, Sequential, Conv2d, BatchNorm2d, LeakyReLU

filters_multiplier = 32
negative_slope = 0.1

bottleneck = {
    "kernel_size": 1,
    "stride": 1,
    "padding": 0,
    "bias": False
}

down_sample = {
    "kernel_size": 3,
    "stride": 2,
    "padding": 1,
    "bias": False
}

casual = {
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "bias": False
}

prelude = {
    "kernel_size": 1,
    "stride": 1,
    "padding": 0,
    "bias": True
}


class Darknet(nn.Module):
    def __init__(self):
        super(Darknet, self).__init__()
        self.intro = Sequential(Conv2d(3, 2 ** 0 * filters_multiplier, **casual),
                                BatchNorm2d(filters_multiplier),
                                LeakyReLU(negative_slope))
        self.module_list = ModuleList([
            ModuleList(
                [Sequential(Conv2d(2 ** i * filters_multiplier, 2 ** (i + 1) * filters_multiplier, **down_sample),
                            BatchNorm2d(2 ** (i + 1) * filters_multiplier),
                            LeakyReLU(negative_slope))] +
                [Sequential(Conv2d(2 ** (i + 1) * filters_multiplier, 2 ** i * filters_multiplier, **bottleneck),
                            BatchNorm2d(2 ** i * filters_multiplier),
                            LeakyReLU(negative_slope),
                            Conv2d(2 ** i * filters_multiplier, 2 ** (i + 1) * filters_multiplier, **casual),
                            BatchNorm2d(2 ** (i + 1) * filters_multiplier),
                            LeakyReLU(negative_slope))
                 ] * num_of_repetitions) for i, num_of_repetitions in enumerate([1, 2, 8, 8, 4])
        ])

    def forward(self, tensor_image):
        tensor = self.intro(tensor_image)
        outs = []
        for i, num_of_repetitions in enumerate([1, 2, 8, 8, 4]):
            tensor = self.module_list[i][0](tensor)
            for j in range(num_of_repetitions):
                tensor += self.module_list[i][j + 1](tensor)
            outs.append(tensor)
        return outs[-3:]


class Tail(nn.Module):
    def __init__(self, number_of_classes, anchors_dims):
        super(Tail, self).__init__()
        self.num_of_yolo_layers = 3
        route_streams = [0, 2 ** 3, 2 ** 2]
        self.harmonics = ModuleList([Sequential(
            Conv2d((2 ** (5 - i) + route_streams[i]) * filters_multiplier, 2 ** (4 - i) * filters_multiplier,
                   **bottleneck),
            BatchNorm2d(2 ** (4 - i) * filters_multiplier),
            LeakyReLU(negative_slope),
            Conv2d(2 ** (4 - i) * filters_multiplier, 2 ** (5 - i) * filters_multiplier, **casual),
            BatchNorm2d(2 ** (5 - i) * filters_multiplier),
            LeakyReLU(negative_slope),
            Conv2d(2 ** (5 - i) * filters_multiplier,
                   2 ** (4 - i) * filters_multiplier, **bottleneck),
            BatchNorm2d(2 ** (4 - i) * filters_multiplier),
            LeakyReLU(negative_slope),
            Conv2d(2 ** (4 - i) * filters_multiplier, 2 ** (5 - i) * filters_multiplier, **casual),
            BatchNorm2d(2 ** (5 - i) * filters_multiplier),
            LeakyReLU(negative_slope)) for i in range(self.num_of_yolo_layers)])
        self.splitted_harmonic = ModuleList([
            ModuleList([
                Sequential(
                    Conv2d(2 ** (5 - i) * filters_multiplier, 2 ** (4 - i) * filters_multiplier, **bottleneck),
                    BatchNorm2d(2 ** (4 - i) * filters_multiplier),
                    LeakyReLU(negative_slope)),
                Sequential(
                    Conv2d(2 ** (4 - i) * filters_multiplier, 2 ** (5 - i) * filters_multiplier, **casual),
                    BatchNorm2d(2 ** (5 - i) * filters_multiplier),
                    LeakyReLU(negative_slope))]) for i in range(self.num_of_yolo_layers)])
        self.preludes = ModuleList([
            Conv2d(2 ** (5 - i) * filters_multiplier, anchors_dims[i] * (number_of_classes + 5), **prelude)
            for i in range(self.num_of_yolo_layers)])
        self.equalizers_for_routes = ModuleList([
            Sequential(
                Conv2d(2 ** (4 - i) * filters_multiplier, 2 ** (3 - i) * filters_multiplier, **bottleneck),
                BatchNorm2d(2 ** (3 - i) * filters_multiplier),
                LeakyReLU(negative_slope))
            for i in range(self.num_of_yolo_layers - 1)])

    def forward(self, routes_hosts):
        out = []
        tensor = routes_hosts[-1]  # 1 x 1024 x 13 x 13
        for i in range(self.num_of_yolo_layers):
            tensor = self.harmonics[i](tensor)
            route_host = self.splitted_harmonic[i][0](tensor)
            tensor = self.splitted_harmonic[i][1](route_host)
            out.append(self.preludes[i](tensor))
            if i < 2:
                tensor = interpolate(self.equalizers_for_routes[i](route_host), scale_factor=2)
                tensor = cat((tensor, routes_hosts[-2 - i]), 1)
        return out


class Head(nn.Module):
    def __init__(self, anchors, number_of_classes=1):
        super(Head, self).__init__()
        self.number_of_classes = number_of_classes
        self.anchors = anchors.view(3, 1, 1, 2)

    def forward(self, features):
        grid_size = Tensor(features.size()[-2:])
        cells_offsets_x, cells_offsets_y = stack(meshgrid(linspace(0, 1, grid_size[0]),
                                                          linspace(0, 1, grid_size[1]).t()), -1)
        features = features.view([-1, len(self.anchors), self.number_of_classes + 5, grid_size[1], grid_size[2]]) \
            .permute(0, 1, 3, 4, 2) \
            .contiguous()
        centers = sigmoid(features[..., :2]) / grid_size + cells_offsets_x
        sizes = exp(features[..., 2:4]) * self.anchors
        probabilities = sigmoid(features[..., 4:])
        return centers, sizes, probabilities

# def bbox_iou(box1, box2, x1y1x2y2=True):
#     """
#     Returns the IoU of two bounding boxes
#     """
#     if not x1y1x2y2:
#         # Transform from center and width to exact coordinates
#         b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
#         b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
#         b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
#         b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
#     else:
#         # Get the coordinates of bounding boxes
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
#
#     # get the corrdinates of the intersection rectangle
#     inter_rect_x1 = torch.max(b1_x1, b2_x1)
#     inter_rect_y1 = torch.max(b1_y1, b2_y1)
#     inter_rect_x2 = torch.min(b1_x2, b2_x2)
#     inter_rect_y2 = torch.min(b1_y2, b2_y2)
#     # Intersection area
#     inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
#         inter_rect_y2 - inter_rect_y1 + 1, min=0
#     )
#     # Union Area
#     b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
#     b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
#
#     iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
#
#     return iou
#
#
# def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
#     """
#     Removes detections with lower object confidence score than 'conf_thres' and performs
#     Non-Maximum Suppression to further filter detections.
#     Returns detections with shape:
#         (x1, y1, x2, y2, object_conf, class_score, class_pred)
#     """
#
#     # From (center x, center y, width, height) to (x1, y1, x2, y2)
#     prediction[..., :4] = xywh2xyxy(prediction[..., :4])
#     output = [None for _ in range(len(prediction))]
#     for image_i, image_pred in enumerate(prediction):
#         # Filter out confidence scores below threshold
#         image_pred = image_pred[image_pred[:, 4] >= conf_thres]
#         # If none are remaining => process next image
#         if not image_pred.size(0):
#             continue
#         # Object confidence times class confidence
#         score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
#         # Sort by it
#         image_pred = image_pred[(-score).argsort()]
#         class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
#         detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
#         # Perform non-maximum suppression
#         keep_boxes = []
#         while detections.size(0):
#             large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
#             label_match = detections[0, -1] == detections[:, -1]
#             # Indices of boxes with lower confidence scores, large IOUs and matching labels
#             invalid = large_overlap & label_match
#             weights = detections[invalid, 4:5]
#             # Merge overlapping bboxes by order of confidence
#             detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
#             keep_boxes += [detections[0]]
#             detections = detections[~invalid]
#         if keep_boxes:
#             output[image_i] = torch.stack(keep_boxes)
#     return output
