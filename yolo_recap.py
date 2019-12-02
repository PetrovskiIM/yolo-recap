import torch
from torch import Tensor, cat
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.nn import ModuleList, Sequential, Conv2d, BatchNorm2d, LeakyReLU

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


class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
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
