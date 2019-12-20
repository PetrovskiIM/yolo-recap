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


class Darknet(Module):
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


class Tail(Module):
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
        tensor = routes_hosts[-1]
        for i in range(self.num_of_yolo_layers):
            tensor = self.harmonics[i](tensor)
            route_host = self.splitted_harmonic[i][0](tensor)
            tensor = self.splitted_harmonic[i][1](route_host)
            out.append(self.preludes[i](tensor))
            if i < 2:
                tensor = interpolate(self.equalizers_for_routes[i](route_host), scale_factor=2)
                tensor = cat((tensor, routes_hosts[-2 - i]), 1)
        return out


class Head(Module):
    def __init__(self, anchors, number_of_classes=1):
        super(Head, self).__init__()
        self.number_of_classes = number_of_classes
        self.anchors = anchors.view(3, 1, 1, 2)

    def forward(self, features):
        grid_size = list(features.size()[-2:])
        cells_offsets = stack(meshgrid(linspace(0, 1, grid_size[0]),
                                       linspace(0, 1, grid_size[1]).t()), -1)
        features = features.view([-1, len(self.anchors), self.number_of_classes + 5] + grid_size) \
            .permute(0, 1, 3, 4, 2) \
            .contiguous()
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
