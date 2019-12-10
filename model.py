import torch
from torch import Tensor, cat, sigmoid, exp
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.nn import ModuleList, Sequential, Conv2d, BatchNorm2d, LeakyReLU
from config import number_of_classes, anchors

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
    def __init__(self, anchors, number_of_classes=1, network_shape=[416, 416]):
        super(Head, self).__init__()
        self.network_shape = network_shape
        self.number_of_classes = number_of_classes
        self.anchors = anchors

    def forward(self, features):
        grid_size = features.size()[-2:]
        stride = self.network_shape[0] / grid_size[0]

        grid = 3
#
#     def forward(self, tensor):
#         # batch_size x S x S x (B*5 +C) = 255
#         batch_size = tensor.size(0)
#         grid_size = tensor.size(2)
#
#         #self.anchors = self.anchors.to(x.device).float()
#         # Calculate offsets for each grid
#         #grid_tensor = torch.arange(num_grid, dtype=torch.float, device=x.device).repeat(num_grid, 1)
#
#         grid_x = grid_tensor.view([1, 1, grid_size, grid_size])
#         grid_y = grid_tensor.t().view([1, 1, grid_size, grid_size])
#
#         #anchor_w = self.anchors[:, 0:1].view((1, -1, 1, 1))
#         #anchor_h = self.anchors[:, 1:2].view((1, -1, 1, 1))
#
#         # Get outputs
#         x_center_pred = sigmoid(tensor[..., 0]) + grid_x #* self.stride  # Center x
#         y_center_pred = sigmoid(tensor[..., 1]) + grid_y #* self.stride  # Center y
#
#         w_pred = exp(tensor[..., 2]) #* anchor_w  # Width
#         h_pred = exp(tensor[..., 3]) #* anchor_h  # Height
#
#         bbox_pred = torch.stack((x_center_pred,
#                                  y_center_pred,
#                                  w_pred,
#                                  h_pred), dim=4).view((num_batch, -1, 4))  # cxcywh
#         conf_pred = sigmoid(prediction_raw[..., 4]).view(num_batch, -1, 1)  # Conf
#         class_predictions = sigmoid(prediction_raw[..., 5:]).view(num_batch, -1, number_of_classes)  # Cls pred one-hot.
#
#         return cat((bbox_pred, conf_pred, class_predictions), -1)
#         if targets is None:
#             return output, 0
#         else:
#             iou_scores, \
#             class_mask, \
#             obj_mask, \
#             noobj_mask, \
#             tx, \
#             ty, \
#             tw, \
#             th, \
#             tcls, \
#             tconf = build_targets(
#                 pred_boxes=pred_boxes,
#                 pred_cls=pred_cls,
#                 target=targets,
#                 anchors=self.scaled_anchors,
#                 ignore_thres=self.ignore_thres,
#             )
#
#             # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
#             loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
#             loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
#             loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
#             loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
#             loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
#             loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
#             loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
#             loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
#             total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
#
#             # Metrics
#             cls_acc = 100 * class_mask[obj_mask].mean()
#             conf_obj = pred_conf[obj_mask].mean()
#             conf_noobj = pred_conf[noobj_mask].mean()
#             conf50 = (pred_conf > 0.5).float()
#             iou50 = (iou_scores > 0.5).float()
#             iou75 = (iou_scores > 0.75).float()
#             detected_mask = conf50 * class_mask * tconf
#             precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
#             recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
#             recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)
#
#             self.metrics = {
#                 "loss": to_cpu(total_loss).item(),
#                 "x": to_cpu(loss_x).item(),
#                 "y": to_cpu(loss_y).item(),
#                 "w": to_cpu(loss_w).item(),
#                 "h": to_cpu(loss_h).item(),
#                 "conf": to_cpu(loss_conf).item(),
#                 "cls": to_cpu(loss_cls).item(),
#                 "cls_acc": to_cpu(cls_acc).item(),
#                 "recall50": to_cpu(recall50).item(),
#                 "recall75": to_cpu(recall75).item(),
#                 "precision": to_cpu(precision).item(),
#                 "conf_obj": to_cpu(conf_obj).item(),
#                 "conf_noobj": to_cpu(conf_noobj).item(),
#                 "grid_size": grid_size,
#             }
#
#             return output, total_loss


# class YOLO(nn.Module):
#
#     def __init__(self, nms=0, thresh=0, hier_thresh=0, post=True):
#         super(YOLO, self).__init__()
#         self.darknet = Darknet()
#         self.tail = Tail()
#
#     def forward(self, image):
#         features = self.darknet(image)
#         liage_1, leage_2, leage_3 = self.tail(features)
#
#         return liage_1
