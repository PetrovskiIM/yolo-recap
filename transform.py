from torch import Tensor, cat, sigmoid, exp, meshgrid, linspace, stack
from torch.nn.functional import interpolate
from torch.nn import Module, ModuleList, Sequential, Conv2d, BatchNorm2d, LeakyReLU

grid_size = list(features.size()[-2:])
cells_offsets = stack(meshgrid(linspace(0, 1, grid_size[0]),
                               linspace(0, 1, grid_size[1]).t()), -1)
features = features.view([-1, len(self.anchors), number_of_classes + 5] + grid_size) \
    .permute(0, 1, 3, 4, 2) \
    .contiguous()
centers = sigmoid(features[..., :2]) / Tensor(grid_size) + cells_offsets
sizes = exp(features[..., 2:4]) * self.anchors
probabilities = sigmoid(features[..., 4:])

def shape_target(grids_size, number_of_classes):
    cells_offsets = stack(meshgrid(linspace(0, 1, grid_size[0]),
                                   linspace(0, 1, grid_size[1]).t()), -1)



