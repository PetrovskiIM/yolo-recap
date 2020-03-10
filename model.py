import numpy as np
import torch
from torch import Tensor, cat, sigmoid, exp, meshgrid, linspace, stack
from torch.nn.functional import interpolate
from torch.nn import Module, ModuleList, Sequential, Conv2d, BatchNorm2d, LeakyReLU

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


class Module(Module):
    def load_weights(self, flatten_weight):
        print("1")


def select_weights(cell, flatten_weights):
    size_expected_by_model = cell.size()
    length_expected_by_model = np.prod(size_expected_by_model)
    return torch.from_numpy(flatten_weights[:length_expected_by_model]).view(size_expected_by_model),\
           flatten_weights[length_expected_by_model:]


def distribute_weights(module, flatten_weight, with_normalization=False):
    state = module.state_dict()
    if with_normalization:
        state["0.bias"], flatten_weight = select_weights(state["0.bias"], flatten_weight)
    else:
        state["0.bias"], flatten_weight = select_weights(state["0.bias"], flatten_weight)
        state["0.weight"], flatten_weight = select_weights(state["0.weight"], flatten_weight)
        state["0.running_mean"], flatten_weight = select_weights(state["0.running_mean"], flatten_weight)
        state["0.running_var"], flatten_weight = select_weights(state["0.running_var"], flatten_weight)
    state["0.weight"], flatten_weight = select_weights(state["0.weight"], flatten_weight)
    module.load_state_dict(state)
    return flatten_weight


class Darknet(Module):
    def __init__(self):
        super(Darknet, self).__init__()
        self.intro = Sequential(Conv2d(3, 2 ** 0 * filters_multiplier, **casual),
                                BatchNorm2d(2 ** 0 * filters_multiplier),
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
                            LeakyReLU(negative_slope)) for _ in range(num_of_repetitions)]
            ) for i, num_of_repetitions in enumerate([1, 2, 8, 8, 4])
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

    def load_weights(self, flatten_weight):
        flatten_weight = distribute_weights(self.intro, flatten_weight)
        for i, num_of_repetitins in enumerate([1, 2, 8, 8, 4]):
            flatten_weight = distribute_weights(self.module_list[i][0], flatten_weight)
            for j in range(num_of_repetitins):
                flatten_weight = distribute_weights(self.module_list[i][j + 1], flatten_weight)


class Tails(Module):
    def __init__(self, number_of_classes, anchors_dims):
        super(Tails, self).__init__()
        self.num_of_yolo_layers = 3
        route_streams = [0, 2 ** 3, 2 ** 2]
        self.tails = ModuleList([
            ModuleList(
                [Sequential(
                    Conv2d((2 ** (5 - i) + route_streams[i]) * filters_multiplier,
                           2 ** (4 - i) * filters_multiplier, **bottleneck),
                    BatchNorm2d(2 ** (4 - i) * filters_multiplier),
                    LeakyReLU(negative_slope),
                    Conv2d(2 ** (4 - i) * filters_multiplier,
                           2 ** (5 - i) * filters_multiplier, **casual),
                    BatchNorm2d(2 ** (5 - i) * filters_multiplier),
                    LeakyReLU(negative_slope),
                    Conv2d(2 ** (5 - i) * filters_multiplier,
                           2 ** (4 - i) * filters_multiplier, **bottleneck),
                    BatchNorm2d(2 ** (4 - i) * filters_multiplier),
                    LeakyReLU(negative_slope),
                    Conv2d(2 ** (4 - i) * filters_multiplier,
                           2 ** (5 - i) * filters_multiplier, **casual),
                    BatchNorm2d(2 ** (5 - i) * filters_multiplier),
                    LeakyReLU(negative_slope)),
                    Sequential(
                        Conv2d(2 ** (5 - i) * filters_multiplier, 2 ** (4 - i) * filters_multiplier, **bottleneck),
                        BatchNorm2d(2 ** (4 - i) * filters_multiplier),
                        LeakyReLU(negative_slope)),
                    Sequential(
                        Conv2d(2 ** (4 - i) * filters_multiplier, 2 ** (5 - i) * filters_multiplier, **casual),
                        BatchNorm2d(2 ** (5 - i) * filters_multiplier),
                        LeakyReLU(negative_slope)),
                    Sequential(Conv2d(2 ** (5 - i) * filters_multiplier, anchors_dims[i] * (number_of_classes + 5),
                                      **prelude))]
                +
                [Sequential(
                    Conv2d(2 ** (4 - i) * filters_multiplier, 2 ** (3 - i) * filters_multiplier, **bottleneck),
                    BatchNorm2d(2 ** (3 - i) * filters_multiplier),
                    LeakyReLU(negative_slope))] * (i < 2)
            ) for i in range(self.num_of_yolo_layers)])

    def forward(self, routes_hosts):
        out = []
        tensor = routes_hosts[-1]
        for i in range(self.num_of_yolo_layers):
            tensor = self.tails[i][0](tensor)
            route_host = self.tails[i][1](tensor)
            tensor = self.tails[i][2](route_host)
            out.append(self.tails[i][3](tensor))
            if i < 2:
                tensor = interpolate(self.tails[i][4](route_host), scale_factor=2, mode="nearest")
                tensor = cat((tensor, routes_hosts[-2 - i]), 1)
        return out

    def load_weight(self, flatten_weight):
        for i in range(self.num_of_yolo_layers):
            flatten_weight = distribute_weights(self.tails[i][0], flatten_weight)
            flatten_weight = distribute_weights(self.tails[i][1], flatten_weight)
            flatten_weight = distribute_weights(self.tails[i][2], flatten_weight)
            flatten_weight = distribute_weights(self.tails[i][3], flatten_weight)
            if i < 2:
                flatten_weight = distribute_weights(self.tails[i][4], flatten_weight)


class Tail(Module):
    def __init__(self, number_of_classes, anchors_dims):
        super(Tail, self).__init__()
        self.num_of_yolo_layers = 3
        route_streams = [0, 2 ** 3, 2 ** 2]
        self.harmonics = ModuleList([Sequential(
            Conv2d((2 ** (5 - i) + route_streams[i]) * filters_multiplier,
                   2 ** (4 - i) * filters_multiplier, **bottleneck),
            BatchNorm2d(2 ** (4 - i) * filters_multiplier),
            LeakyReLU(negative_slope),
            Conv2d(2 ** (4 - i) * filters_multiplier,
                   2 ** (5 - i) * filters_multiplier, **casual),
            BatchNorm2d(2 ** (5 - i) * filters_multiplier),
            LeakyReLU(negative_slope),
            Conv2d(2 ** (5 - i) * filters_multiplier,
                   2 ** (4 - i) * filters_multiplier, **bottleneck),
            BatchNorm2d(2 ** (4 - i) * filters_multiplier),
            LeakyReLU(negative_slope),
            Conv2d(2 ** (4 - i) * filters_multiplier,
                   2 ** (5 - i) * filters_multiplier, **casual),
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
        tensor = routes_hosts[-1]
        for i in range(self.num_of_yolo_layers):
            tensor = self.harmonics[i](tensor)
            route_host = self.splitted_harmonic[i][0](tensor)
            tensor = self.splitted_harmonic[i][1](route_host)
            out.append(self.preludes[i](tensor))
            if i < 2:
                tensor = interpolate(self.equalizers_for_routes[i](route_host), scale_factor=2, mode="nearest")
                tensor = cat((tensor, routes_hosts[-2 - i]), 1)
        return out


class Head(Module):
    def __init__(self, anchors, number_of_classes=1):
        super(Head, self).__init__()
        self.number_of_classes = number_of_classes
        self.anchors = anchors.view(3, 1, 1, 2)

    def forward(self, features):
        grid_size = list(features.size()[-2:])
        cells_offsets = stack(meshgrid(linspace(0, 1 - 1 / grid_size[0], grid_size[0]),
                                       linspace(0, 1 - 1 / grid_size[1], grid_size[1])), -1)[..., [1, 0]]
        features = features.view([-1, len(self.anchors), self.number_of_classes + 5] + grid_size) \
            .permute(0, 1, 3, 4, 2) \
            # .contiguous()
        centers = sigmoid(features[..., :2]) / Tensor(grid_size) + cells_offsets
        sizes = exp(features[..., 2:4]) * self.anchors
        probabilities = sigmoid(features[..., 4:])
        return centers, sizes, probabilities


class Yolo(Module):
    def __init__(self, anchors, anchors_dim, number_of_classes=1):
        super(Yolo, self).__init__()
        self.feature_extractor = Darknet()
        self.tail = Tail(number_of_classes, anchors_dim)
        self.head = Head(anchors, number_of_classes)

    def forward(self, image):
        features = self.feature_extractor(image)
        return self.head(self.tail(features)[0])
