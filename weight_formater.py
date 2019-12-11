import numpy as np
import torch
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from model import Darknet, Tail

weight_file_path = "/home/ivan/sets/three_days/backup/v3single_1256008.weights"
# Open the weights file
with open(weight_file_path, "rb") as f:
    header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
    header_info = header  # Needed to write header when saving weights
    seen = header[3]  # number of images seen during training
    weights = np.fromfile(f, dtype=np.float32)  # The rest are weights
print(seen)

image = torch.randn([3, 416, 416])
darknet = Darknet()
tail = Tail(1, [3, 3, 3])
initial_detections = tail(darknet(image.unsqueeze(0)))


def parse_darknet_weight(flatten_weights, model):
    checkpoint = model.state_dict()
    new_dict = checkpoint.copy()
    for param_tensor in checkpoint:
        if param_tensor.endswith("weight") or \
                param_tensor.endswith("running_mean") or \
                param_tensor.endswith("running_var") or \
                param_tensor.endswith("bias"):
            size_expected_by_model = checkpoint[param_tensor].size()

            length_expected_by_model = np.prod(size_expected_by_model)
            new_dict[param_tensor] = \
                torch.from_numpy(flatten_weights[:length_expected_by_model]).view(size_expected_by_model)
            flatten_weights = flatten_weights[length_expected_by_model:]
    return new_dict, flatten_weights


darknet_state_dict, unused_weights = parse_darknet_weight(weights, darknet)
tail_state_dict, unused_weights = parse_darknet_weight(unused_weights, tail)

darknet = darknet.load_state_dict(darknet_state_dict)
tail = tail.load_state_dict(tail_state_dict)


print(unused_weights.shape)
#print(initial_detections[0] == tail(darknet(image.unsqueeze(0)))[0])

