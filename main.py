import torch
import pandas as pd
from torch import Tensor, cat, sigmoid, exp, stack, max
import matplotlib
import cv2

image = torch.randn([3, 416, 416])

from model import Darknet, Tail, Head

image = cv2.imread("/home/ivan/Projects/yolo-recap/ee.jpg")
image = cv2.resize(image, (416, 416), interpolation=cv2.INTER_LINEAR)
image = torch.Tensor(image).permute(2, 0, 1)

import numpy as np
import torch

from model import Darknet, Tail

weight_file_path = "/home/ivan/Projects/yolodbmanager/weights/smallcars/production/yolo.weights"
with open(weight_file_path, "rb") as f:
    header = np.fromfile(f, dtype=np.int32, count=5)
    header_info = header
    seen = header[3]
    weights = np.fromfile(f, dtype=np.float32)
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

darknet.load_state_dict(darknet_state_dict)
tail.load_state_dict(tail_state_dict)

anchors = Tensor([[13. / 416, 16. / 416], [11. / 416, 35. / 416], [22. / 416, 24. / 416]])
anchors_width = anchors[:, 0].view((1, len(anchors), 1, 1))
anchors_height = anchors[:, 1].view((1, len(anchors), 1, 1))
head = Head(anchors)
lines = head(tail(darknet(image.unsqueeze(0)))[0]).detach().numpy()[0]
# lines = 3
ll = pd.DataFrame(lines)
ll[0] = 0
ll.sample(frac=1).to_csv("./ee.txt", index=None, header=None, mode="a+", sep=" ")
lines = head(tail(darknet(image.unsqueeze(0)))[1]).detach().numpy()[0]
# lines = 3
ll = pd.DataFrame(lines)
ll[0] = 0
ll.sample(frac=1).to_csv("./ee.txt", index=None, header=None, mode="a", sep=" ")
lines = head(tail(darknet(image.unsqueeze(0)))[2]).detach().numpy()[0]
# lines = 3
ll = pd.DataFrame(lines)
ll[0] = 0
ll.sample(frac=1).to_csv("./ee.txt", index=None, header=None, mode="a", sep=" ")
