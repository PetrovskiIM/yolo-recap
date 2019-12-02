import torch
from torch import Tensor
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

        for i, num_of_repetitions in enumerate([2, 8, 8, 4]):
            for j in range(num_of_repetitions):
                tensor += self.module_list[i][j](tensor)
        return tensor


class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        self.backbone = Darknet()
        num_of_yolo_layers = 3
        self.harmonics = ModuleList([
            ModuleList(
                [Sequential(Conv2d(2 ** (5 - i) * filters_multiplier, 2 ** (4 - i) * filters_multiplier, **bottleneck),
                            BatchNorm2d(2 ** (4 - i) * filters_multiplier),
                            LeakyReLU(negative_slope),
                            Conv2d(2 ** (4 - i) * filters_multiplier, 2 ** (5 - i) * filters_multiplier, **casual),
                            BatchNorm2d(2 ** (5 - i) * filters_multiplier),
                            LeakyReLU(negative_slope))] * 3) for i in range(num_of_yolo_layers)])

        self.prelude1 = self.Sequential(Conv2d(2 ** 4 * filters_multiplier, 255, **bottleneck),
                                        BatchNorm2d(255))
        # self.prelude2 = self.Sequential(Conv2d(!!!!, 255, ** bottleneck)

    def forward(self, tensor):
        darknet_features = self.darknet(tensor)

        tensor = self.harmonics[0][0](darknet_features)
        route_host = self.harmonics[0][1](tensor)
        tensor = self.harmonics[0][2](route_host)
        tensor = self.prelude1(tensor)
        self.yolo()

        tensor = self.harmonics[1][0](darknet_features)
        route_host = self.harmonics[1][1](tensor)
        tensor = self.harmonics[1][2](route_host)
        tensor = self.prelude2(tensor)
        self.yolo()
        tensor = self.harmonics[2][0](darknet_features)
        route_host = self.harmonics[2][1](tensor)
        tensor = self.harmonics[2][2](route_host)
        tensor = self.prelude2(tensor)
        self.yolo()
        return x
