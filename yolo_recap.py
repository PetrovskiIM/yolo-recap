import torch
from torch import Tensor, cat
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.nn import ModuleList, Sequential, Conv2d, BatchNorm2d, LeakyReLU
from config import number_of_classes, anchors

filters_multiplier = 32
negative_slope = 0.1

bottleneck = {
    "kernel_size": 1,
    "stride": 1,
    "padding": 1,
    "bias": False
}

downsample = {
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
        self.into = Sequential(Conv2d(3, 2 ** 0 * filters_multiplier, **casual),
                               BatchNorm2d(filters_multiplier),
                               LeakyReLU(negative_slope))
        self.module_list = ModuleList([
            ModuleList(
                [Sequential(Conv2d(2 ** (i + 1) * filters_multiplier, 2 ** (i + 2) * filters_multiplier, **downsample),
                            BatchNorm2d(2 ** (i + 2) * filters_multiplier),
                            LeakyReLU(negative_slope))] +
                [Sequential(Conv2d(2 ** (i + 2) * filters_multiplier, 2 ** (i + 1) * filters_multiplier, **bottleneck),
                            BatchNorm2d(2 ** (i + 1) * filters_multiplier),
                            LeakyReLU(negative_slope),
                            Conv2d(2 ** (i + 1) * filters_multiplier, 2 ** (i + 2) * filters_multiplier, **casual),
                            BatchNorm2d(2 ** (i + 2) * filters_multiplier),
                            LeakyReLU(negative_slope))
                 ] * num_of_repetitions) for i, num_of_repetitions in enumerate([2, 8, 8, 4])
        ])

    def forward(self, tensor_image):
        tensor = self.into(tensor_image)
        outs = []
        for i, num_of_repetitions in enumerate([2, 8, 8, 4]):
            for j in range(num_of_repetitions):
                tensor += self.module_list[i][j](tensor)
            outs.append(tensor)
        return outs[-3:]


class Tail(nn.Module):
    def __init__(self):
        super(Tail, self).__init__()
        self.backbone = Darknet()
        self.num_of_yolo_layers = 3
        route_streams = [0, 3, 2]
        self.harmonics = ModuleList([
            ModuleList(
                [Sequential(Conv2d(2 ** (5 - i + route_streams[i]) * filters_multiplier,
                                   2 ** (4 - i) * filters_multiplier, **bottleneck),
                            BatchNorm2d(2 ** (4 - i) * filters_multiplier),
                            LeakyReLU(negative_slope),
                            Conv2d(2 ** (4 - i) * filters_multiplier, 2 ** (5 - i) * filters_multiplier, **casual),
                            BatchNorm2d(2 ** (5 - i) * filters_multiplier),
                            LeakyReLU(negative_slope))] * 3) for i in range(self.num_of_yolo_layers)])
        self.preludes = ModuleList([
            self.Sequential(Conv2d(2 ** (5-i) * filters_multiplier, 255, **bottleneck),
                            BatchNorm2d(255)) for i in range(self.num_of_yolo_layers)])
        self.equalizers_for_routes = ModuleList([
            interpolate(Sequential(
                Conv2d(2 ** (4-i) * filters_multiplier, 2 ** (3-i) * filters_multiplier, **bottleneck),
                BatchNorm2d(2 ** (3-i) * filters_multiplier),
                LeakyReLU(negative_slope)
            ), scale_factor=2) for i in range(self.num_of_yolo_layers-1)])

    def forward(self, routes_hosts):
        out = []
        tensor = routes_hosts[-1]
        for i in range(self.num_of_yolo_layers - 1):
            tensor = self.harmonics[i][0](tensor)
            route_host = self.harmonics[i][1](tensor)
            tensor = self.harmonics[i][2](route_host)
            out.append(self.preludes[i](tensor))
            tensor = self.equalizers_for_routes[i](route_host)
            tensor = cat(tensor, routes_hosts[-2 - i])
        for j in range(3):
            tensor = self.harmonics[2][j](tensor)
        out.append(self.prelude[2](tensor))
        return out


class Head(nn.Module):
    def __init__(self, scale, stride, anchors):
        super(Head, self).__init__()
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            idx = None
        self.anchors = torch.tensor([anchors[i] for i in idx])
        self.stride = stride

    def forward(self, x):
        num_batch = x.size(0)
        num_grid = x.size(2)

        if self.training:
            output_raw = x.view(num_batch,
                                NUM_ANCHORS_PER_SCALE,
                                NUM_ATTRIB,
                                num_grid,
                                num_grid).permute(0, 1, 3, 4, 2).contiguous().view(num_batch, -1, NUM_ATTRIB)
            return output_raw
        else:
            prediction_raw = x.view(num_batch,
                                    NUM_ANCHORS_PER_SCALE,
                                    NUM_ATTRIB,
                                    num_grid,
                                    num_grid).permute(0, 1, 3, 4, 2).contiguous()

            self.anchors = self.anchors.to(x.device).float()
            # Calculate offsets for each grid
            grid_tensor = torch.arange(num_grid, dtype=torch.float, device=x.device).repeat(num_grid, 1)
            grid_x = grid_tensor.view([1, 1, num_grid, num_grid])
            grid_y = grid_tensor.t().view([1, 1, num_grid, num_grid])
            anchor_w = self.anchors[:, 0:1].view((1, -1, 1, 1))
            anchor_h = self.anchors[:, 1:2].view((1, -1, 1, 1))

            # Get outputs
            x_center_pred = (torch.sigmoid(prediction_raw[..., 0]) + grid_x) * self.stride # Center x
            y_center_pred = (torch.sigmoid(prediction_raw[..., 1]) + grid_y) * self.stride  # Center y

            w_pred = torch.exp(prediction_raw[..., 2]) * anchor_w  # Width
            h_pred = torch.exp(prediction_raw[..., 3]) * anchor_h  # Height

            bbox_pred = torch.stack((x_center_pred,
                                     y_center_pred,
                                     w_pred,
                                     h_pred), dim=4).view((num_batch, -1, 4)) #cxcywh
            conf_pred = torch.sigmoid(prediction_raw[..., 4]).view(num_batch, -1, 1)  # Conf
            class_predictions = torch.sigmoid(prediction_raw[..., 5:]).view(num_batch, -1, number_of_classes)  # Cls pred one-hot.

            return cat((bbox_pred, conf_pred, class_predictions), -1)


# class YOLO(nn.Module):
#
#     def __init__(self, nms=False, post=True):
#         super(YoloNetV3, self).__init__()
#         self.darknet = DarkNet53BackBone()
#         self.yolo_tail = YoloNetTail()
#         self.nms = nms
#         self._post_process = post
#
#     def forward(self, x):
#         tmp1, tmp2, tmp3 = self.darknet(x)
#         out1, out2, out3 = self.yolo_tail(tmp1, tmp2, tmp3)
#         out = torch.cat((out1, out2, out3), 1)
#         logging.debug("The dimension of the output before nms is {}".format(out.size()))
#         return out