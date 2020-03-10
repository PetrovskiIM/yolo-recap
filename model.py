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


def select_weights(cell, flatten_weights):
    size_expected_by_model = cell.size()
    length_expected_by_model = np.prod(size_expected_by_model)
    return torch.from_numpy(flatten_weights[:length_expected_by_model]).view(size_expected_by_model), \
           flatten_weights[length_expected_by_model:]


def distribute_weights(module, flatten_weight, number_of_layers_in_group=1, with_normalization=True):
    state = module.state_dict()
    for i in range(number_of_layers_in_group):
        if not with_normalization:
            state[f"{3 * i}.bias"], flatten_weight = select_weights(state[f"{3 * i}.bias"], flatten_weight)
        else:
            for ending in ["bias", "weight", "running_mean", "running_var"]:
                state[f"{3 * i + 1}.{ending}"], flatten_weight = select_weights(state[f"{3 * i + 1}.{ending}"],
                                                                                flatten_weight)
        state[f"{3 * i}.weight"], flatten_weight = select_weights(state[f"{3 * i}.weight"], flatten_weight)
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
        for i, num_of_repetitions in enumerate([1, 2, 8, 8, 4]):
            flatten_weight = distribute_weights(self.module_list[i][0], flatten_weight)
            for j in range(num_of_repetitions):
                flatten_weight = distribute_weights(self.module_list[i][j + 1], flatten_weight, 2)
        return flatten_weight


class Tail(Module):
    def __init__(self, number_of_classes, anchors_dims):
        super(Tail, self).__init__()
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

    def load_weights(self, flatten_weight):
        for i in range(self.num_of_yolo_layers):
            flatten_weight = distribute_weights(self.tails[i][0], flatten_weight, 4)
            flatten_weight = distribute_weights(self.tails[i][1], flatten_weight)
            flatten_weight = distribute_weights(self.tails[i][2], flatten_weight)
            flatten_weight = distribute_weights(self.tails[i][3], flatten_weight, with_normalization=False)
            if i < 2:
                flatten_weight = distribute_weights(self.tails[i][4], flatten_weight)


class Head(Module):
    def __init__(self, anchors, number_of_classes=1):
        super(Head, self).__init__()
        self.number_of_classes = number_of_classes
        self.anchors = anchors.view(3, 1, 1, 2)

    def forward(self, features):
        grid_size = list(features.size()[-2:])
        cells_offsets = stack(meshgrid(linspace(0, 1 - 1 / grid_size[0], grid_size[0]),
                                       linspace(0, 1 - 1 / grid_size[1], grid_size[1])), -1)[..., [1, 0]]
        features = features.view([-1, len(self.anchors), self.number_of_classes + 5] + grid_size).permute(0, 1, 3, 4, 2)
        centers = sigmoid(features[..., :2]) / Tensor(grid_size) + cells_offsets
        sizes = exp(features[..., 2:4]) * self.anchors
        probabilities = sigmoid(features[..., 4:])
        return centers, sizes, probabilities


class Yolo(Module):
    def __init__(self, anchors, anchors_dim, number_of_classes=1):
        super(Yolo, self).__init__()
        self.number_of_classes = number_of_classes
        self.feature_extractor = Darknet()
        self.tail = Tail(number_of_classes, anchors_dim)
        self.heads = [Head(anchors[i], number_of_classes) for i in range(3)]
        print(anchors[0])

    def load_weights(self, flatten_weight):
        self.tail.load_weights(self.feature_extractor.load_weights(flatten_weight))

    def forward(self, image):
        features = self.feature_extractor(image)
        print(features[0].size())
        print(features[1].size())
        print(features[2].size())
        centers_0, sizes_0, probabilities_0 = self.heads[0](features[0])
        centers_1, sizes_1, probabilities_1 = self.heads[1](features[1])
        centers_2, sizes_2, probabilities_2 = self.heads[2](features[2])
        return {
            "boxes": stack((cat((centers_0, sizes_0), -1).view(-1, 4),
                            cat((centers_1, sizes_1), -1).view(-1, 4),
                            cat((centers_2, sizes_2), -1).view(-1, 4))),
            "scores": stack((probabilities_0.view(-1, 1 + self.number_of_classes),
                             probabilities_1.view(-1, 1 + self.number_of_classes),
                             probabilities_2.view(-1, 1 + self.number_of_classes)))
        }