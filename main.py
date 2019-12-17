import numpy as np
import pandas as pd
import matplotlib
import cv2

import torch
from torch import Tensor, cat, sigmoid, exp, stack, max
from model import Darknet, Tail, Head

image = cv2.imread("/home/ivan/Projects/yolo-recap/ee.jpg")
image = cv2.resize(image, (416, 416), interpolation=cv2.INTER_LINEAR)
image = torch.Tensor(image).permute(2, 0, 1)

weight_file_path = "/home/ivan/Projects/yolodbmanager/weights/smallcars/production/yolo.weights"
with open(weight_file_path, "rb") as f:
    header = np.fromfile(f, dtype=np.int32, count=5)
    header_info = header
    seen = header[3]
    weights = np.fromfile(f, dtype=np.float32)
print(seen)

darknet = Darknet()
tail = Tail(1, [3, 3, 3])


def parse_darknet_weight(flatten_weights, model):
    checkpoint = model.state_dict()
    new_dict = checkpoint.copy()
    endings = ["bias", "weight", "running_mean", "running_var"]
    previous = ""
    for layer in checkpoint:
        if previous != ".".join(layer.split(".")[:-1]):
            previous = ".".join(layer.split(".")[:-1])
            if layer.split(".")[-1] in ["bias", "weight", "running_mean", "running_var"]:
                unordered_keys = [".".join(layer.split(".")[:-1]) + "." + ending for ending in endings]
                if (unordered_keys[0] in checkpoint.keys()) & \
                        (unordered_keys[1] in checkpoint.keys()) & \
                        (unordered_keys[2] in checkpoint.keys()) & \
                        (unordered_keys[3] in checkpoint.keys()):
                    for unordered_key in unordered_keys:
                      #  print(unordered_key)
                        size_expected_by_model = checkpoint[unordered_key].size()
                        length_expected_by_model = np.prod(size_expected_by_model)
                        new_dict[unordered_key] = \
                            torch.from_numpy(flatten_weights[:length_expected_by_model]).view(size_expected_by_model)
                        flatten_weights = flatten_weights[length_expected_by_model:]
                elif (unordered_keys[0] in checkpoint.keys()) & \
                        (unordered_keys[1] in checkpoint.keys()):
                    for unordered_key in unordered_keys[:2]:
                    #    print(unordered_key)
                        size_expected_by_model = checkpoint[unordered_key].size()
                        length_expected_by_model = np.prod(size_expected_by_model)
                        new_dict[unordered_key] = \
                            torch.from_numpy(flatten_weights[:length_expected_by_model]).view(size_expected_by_model)
                        flatten_weights = flatten_weights[length_expected_by_model:]
                elif unordered_keys[1] in checkpoint.keys():
                    unordered_key = unordered_keys[1]
                 #   print(unordered_key)
                    size_expected_by_model = checkpoint[unordered_key].size()
                    length_expected_by_model = np.prod(size_expected_by_model)
                    new_dict[unordered_key] = \
                        torch.from_numpy(flatten_weights[:length_expected_by_model]).view(size_expected_by_model)
                    flatten_weights = flatten_weights[length_expected_by_model:]
    return new_dict, flatten_weights


def parse_tail_weight(flatten_weights, model):
    checkpoint = model.state_dict()
    new_dict = checkpoint.copy()
    endings = ["bias", "weight", "running_mean", "running_var"]
    previous = ""
    for i in range(3):
        for layer_type in ["harmonics", "splitted_harmonic", "preludes", "equalizers_for_routes"]:
            for layer in checkpoint:
                if layer_type in layer:
                    if int(layer.split(".")[1]) == i:
                        if previous != ".".join(layer.split(".")[:-1]):
                            previous = ".".join(layer.split(".")[:-1])
                            if layer.split(".")[-1] in ["bias", "weight", "running_mean", "running_var"]:
                                unordered_keys = [".".join(layer.split(".")[:-1]) + "." + ending for ending in endings]
                                if (unordered_keys[0] in checkpoint.keys()) & \
                                        (unordered_keys[1] in checkpoint.keys()) & \
                                        (unordered_keys[2] in checkpoint.keys()) & \
                                        (unordered_keys[3] in checkpoint.keys()):
                                    for unordered_key in unordered_keys:
                                        print(unordered_key)
                                        size_expected_by_model = checkpoint[unordered_key].size()
                                        length_expected_by_model = np.prod(size_expected_by_model)
                                        new_dict[unordered_key] = \
                                            torch.from_numpy(flatten_weights[:length_expected_by_model]).view(
                                                size_expected_by_model)
                                        flatten_weights = flatten_weights[length_expected_by_model:]
                                elif (unordered_keys[0] in checkpoint.keys()) & \
                                        (unordered_keys[1] in checkpoint.keys()):
                                    for unordered_key in unordered_keys[:2]:
                                        print(unordered_key)
                                        size_expected_by_model = checkpoint[unordered_key].size()
                                        length_expected_by_model = np.prod(size_expected_by_model)
                                        new_dict[unordered_key] = \
                                            torch.from_numpy(flatten_weights[:length_expected_by_model]).view(
                                                size_expected_by_model)
                                        flatten_weights = flatten_weights[length_expected_by_model:]
                                elif unordered_keys[1] in checkpoint.keys():
                                    unordered_key = unordered_keys[1]
                                    print(unordered_key)
                                    size_expected_by_model = checkpoint[unordered_key].size()
                                    length_expected_by_model = np.prod(size_expected_by_model)
                                    new_dict[unordered_key] = \
                                        torch.from_numpy(flatten_weights[:length_expected_by_model]).view(
                                            size_expected_by_model)
                                    flatten_weights = flatten_weights[length_expected_by_model:]
    return new_dict, flatten_weights


darknet_state_dict, unused_weights = parse_darknet_weight(weights, darknet)
tail_state_dict, unused_weights = parse_tail_weight(unused_weights, tail)

# darknet.load_state_dict(darknet_state_dict)
# tail.load_state_dict(tail_state_dict)
#
# anchors = Tensor([[13. / 416, 16. / 416], [11. / 416, 35. / 416], [22. / 416, 24. / 416]])
# head = Head(anchors)
#
# center, sizes, probs = head(tail(darknet(image.unsqueeze(0)))[0]).detach().numpy()[0]
# print(center.size())

# # lines = 3
# ll = pd.DataFrame(lines)
# ll[0] = 0
# ll.sample(frac=1).to_csv("./ee.txt", index=None, header=None, mode="a+", sep=" ")
# lines = head(tail(darknet(image.unsqueeze(0)))[1]).detach().numpy()[0]
# # lines = 3
# ll = pd.DataFrame(lines)
# ll[0] = 0
# ll.sample(frac=1).to_csv("./ee.txt", index=None, header=None, mode="a", sep=" ")
# lines = head(tail(darknet(image.unsqueeze(0)))[2]).detach().numpy()[0]
# # lines = 3
# ll = pd.DataFrame(lines)
# ll[0] = 0
# ll.sample(frac=1).to_csv("./ee.txt", index=None, header=None, mode="a", sep=" ")
