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
    "bias": True
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
    def __init__(self):
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
            Sequential(Conv2d(2 ** (5 - i) * filters_multiplier, 255, **bottleneck),
                       BatchNorm2d(255)) for i in range(self.num_of_yolo_layers)])
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
    def __init__(self, anchors, number_classes, network_width):
         super(YOLO, self).__init__()
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
#
#
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
