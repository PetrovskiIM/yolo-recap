import numpy as np
import torch
from model import Darknet, Tail

weight_file_path = "/home/ivan/sets/best_big/backup/v3single_last.weights"
with open(weight_file_path, "rb") as f:
    header = np.fromfile(f, dtype=np.int32, count=5)
    header_info = header
    seen = header[3]
    weights = np.fromfile(f, dtype=np.float32)

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
                if all(unordered_key in checkpoint.keys() for unordered_key in unordered_keys):
                    for unordered_key in unordered_keys:
                        size_expected_by_model = checkpoint[unordered_key].size()
                        length_expected_by_model = np.prod(size_expected_by_model)
                        new_dict[unordered_key] = \
                            torch.from_numpy(flatten_weights[:length_expected_by_model]).view(size_expected_by_model)
                        flatten_weights = flatten_weights[length_expected_by_model:]
                    chunks = unordered_key.split('.')
                    chunks[-2] = str(int(chunks[-2]) - 1)
                    unordered_key = ".".join(chunks[:-1]) + ".weight"
                    size_expected_by_model = checkpoint[unordered_key].size()
                    length_expected_by_model = np.prod(size_expected_by_model)
                    new_dict[unordered_key] = \
                        torch.from_numpy(flatten_weights[:length_expected_by_model]).view(size_expected_by_model)
                    flatten_weights = flatten_weights[length_expected_by_model:]
                elif (unordered_keys[0] in checkpoint.keys()) & (unordered_keys[1] in checkpoint.keys()):
                    for unordered_key in unordered_keys[:2]:
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
                                if all(unordered_key in checkpoint.keys() for unordered_key in unordered_keys):
                                    for unordered_key in unordered_keys:
                                        size_expected_by_model = checkpoint[unordered_key].size()
                                        length_expected_by_model = np.prod(size_expected_by_model)
                                        new_dict[unordered_key] = \
                                            torch.from_numpy(flatten_weights[:length_expected_by_model]).view(
                                                size_expected_by_model)
                                        flatten_weights = flatten_weights[length_expected_by_model:]
                                    chunks = unordered_key.split('.')
                                    chunks[-2] = str(int(chunks[-2]) - 1)
                                    unordered_key = ".".join(chunks[:-1]) + ".weight"
                                    size_expected_by_model = checkpoint[unordered_key].size()
                                    length_expected_by_model = np.prod(size_expected_by_model)
                                    new_dict[unordered_key] = \
                                        torch.from_numpy(flatten_weights[:length_expected_by_model]).view(
                                            size_expected_by_model)
                                    flatten_weights = flatten_weights[length_expected_by_model:]
                                elif (unordered_keys[0] in checkpoint.keys()) & \
                                        (unordered_keys[1] in checkpoint.keys()):
                                    for unordered_key in unordered_keys[:2]:
                                        size_expected_by_model = checkpoint[unordered_key].size()
                                        length_expected_by_model = np.prod(size_expected_by_model)
                                        new_dict[unordered_key] = \
                                            torch.from_numpy(flatten_weights[:length_expected_by_model]).view(
                                                size_expected_by_model)
                                        flatten_weights = flatten_weights[length_expected_by_model:]
    return new_dict, flatten_weights

#
# darknet_state_dict, unused_weights = parse_darknet_weight(weights, darknet)
# tail_state_dict, unused_weights = parse_tail_weight(unused_weights, tail)
# darknet.load_state_dict(darknet_state_dict)
# tail.load_state_dict(tail_state_dict)


def reorder_keys(word, repetition_number=None, local_i=0, type_number=None):
    version_ended_by_dot = ""
    if repetition_number is not None:
        version_ended_by_dot += str(repetition_number) + "."

    if type_number is not None:
        version_ended_by_dot += str(type_number) + "."
    d= {
        "convolution_biases": f"{word}.{version_ended_by_dot}{local_i}.bias",
        "convolution_weights": f"{word}.{version_ended_by_dot}{local_i}.weight",
        "weights": f"{word}.{version_ended_by_dot}{local_i + 1}.weight",
        "biases": f"{word}.{version_ended_by_dot}{local_i + 1}.bias",
        "running_mean": f"{word}.{version_ended_by_dot}{local_i + 1}.running_mean",
        "running_var": f"{word}.{version_ended_by_dot}{local_i + 1}.running_var"
    }
    # print(d["weights"])
    # print(d["biases"])
    # print(d["running_mean"])
    # print(d["running_var"])
    # print(d["convolution_weights"])
    return d


def format_darknet():
    reorder_keys("intro")
    for repetition_number, num_of_types in enumerate([1, 2, 8, 8, 4]):
        for type_number in range(num_of_types + 1):
            if type_number == 0:
                reorder_keys("module_list", repetition_number, local_i=0, type_number=type_number)
            else:
                local_i = 0
                reorder_keys("module_list", repetition_number, local_i, type_number)
                local_i += 3
                reorder_keys("module_list", repetition_number, local_i, type_number)


def format_tail(number_of_yolo_layers):
    for repetition_number in range(number_of_yolo_layers):
        local_i = 0
        reorder_keys("harmonics", repetition_number, local_i)
        local_i += 3
        reorder_keys("harmonics", repetition_number, local_i)
        local_i += 3
        reorder_keys("harmonics", repetition_number, local_i)
        local_i += 3
        reorder_keys("harmonics", repetition_number, local_i)

        reorder_keys("splitted_harmonic", repetition_number, type_number=0)
        reorder_keys("splitted_harmonic", repetition_number, type_number=1)

        reorder_keys("preludes", repetition_number)
        reorder_keys("equalizers_for_routes", repetition_number)


def allocate_weight(key, checkpoint, flatten_weights):
    size_expected_by_model = checkpoint[key].size()
    length_expected_by_model = np.prod(size_expected_by_model)
    checkpoint[key] = torch.from_numpy(flatten_weights[:length_expected_by_model]).view(size_expected_by_model)
    return flatten_weights[length_expected_by_model:]


def allocate_weight_layer(darknet_to_pytorch, checkpoint, flatten_weight, with_normalization=True):
    if with_normalization:
        flatten_weight = allocate_weight(darknet_to_pytorch["biases"], checkpoint, flatten_weight)
        flatten_weight = allocate_weight(darknet_to_pytorch["weights"], checkpoint, flatten_weight)
        flatten_weight = allocate_weight(darknet_to_pytorch["running_mean"], checkpoint, flatten_weight)
        flatten_weight = allocate_weight(darknet_to_pytorch["running_var"], checkpoint, flatten_weight)
        flatten_weight = allocate_weight(darknet_to_pytorch["convolution_weights"], checkpoint, flatten_weight)
    else:
        flatten_weight = allocate_weight(darknet_to_pytorch["convolution_biases"], checkpoint, flatten_weight)
        flatten_weight = allocate_weight(darknet_to_pytorch["convolution_weights"], checkpoint, flatten_weight)
    return flatten_weight


def load_checkpoint(darknet, tail, flatten_weight):
    checkpoint = darknet.state_dict().copy()
    number_of_yolo_layers = 3
    flatten_weight = allocate_weight_layer(reorder_keys("intro"), checkpoint, flatten_weight)
    for repetition_number, num_of_types in enumerate([1, 2, 8, 8, 4]):
        for type_number in range(num_of_types + 1):
            if type_number == 0:
                flatten_weight = \
                    allocate_weight_layer(reorder_keys("module_list",
                                                       repetition_number,
                                                       type_number=type_number),
                                          checkpoint,
                                          flatten_weight)
            else:
                local_i = 0
                flatten_weight = \
                    allocate_weight_layer(reorder_keys("module_list",
                                                       repetition_number,
                                                       local_i,
                                                       type_number),
                                          checkpoint,
                                          flatten_weight)
                local_i += 3
                flatten_weight = \
                    allocate_weight_layer(reorder_keys("module_list",
                                                       repetition_number,
                                                       local_i,
                                                       type_number),
                                          checkpoint,
                                          flatten_weight)
    print(darknet.load_state_dict(checkpoint))
    checkpoint = tail.state_dict().copy()
    for repetition_number in range(number_of_yolo_layers):
        #print(repetition_number)
        local_i = 0
        flatten_weight = \
            allocate_weight_layer(reorder_keys("harmonics", repetition_number, local_i),
                                  checkpoint,
                                  flatten_weight)
        local_i += 3
        flatten_weight = \
            allocate_weight_layer(reorder_keys("harmonics", repetition_number, local_i),
                                  checkpoint,
                                  flatten_weight)
        local_i += 3
        flatten_weight = \
            allocate_weight_layer(reorder_keys("harmonics", repetition_number, local_i),
                                  checkpoint,
                                  flatten_weight)
        local_i += 3
        flatten_weight = \
            allocate_weight_layer(reorder_keys("harmonics", repetition_number, local_i),
                                  checkpoint,
                                  flatten_weight)

        flatten_weight = \
            allocate_weight_layer(reorder_keys("splitted_harmonic", repetition_number, type_number=0),
                                  checkpoint,
                                  flatten_weight)
        flatten_weight = \
            allocate_weight_layer(reorder_keys("splitted_harmonic", repetition_number, type_number=1),
                                  checkpoint,
                                  flatten_weight)
        print(len(flatten_weight))
        flatten_weight = \
            allocate_weight_layer({
            "convolution_biases": f"preludes.{repetition_number}.bias",
            "convolution_weights": f"preludes.{repetition_number}.weight"
            },
                                  checkpoint,
                                  flatten_weight, with_normalization=False)
          # flatten_weight = \
        #     allocate_weight_layer(reorder_keys("preludes", repetition_number),
        #                           checkpoint,
        #                           flatten_weight, with_normalization=False)
        if repetition_number<2:
            flatten_weight = \
                allocate_weight_layer(reorder_keys("equalizers_for_routes", repetition_number),
                                      checkpoint,
                                      flatten_weight)
    tail.load_state_dict(checkpoint)
    return darknet, tail, flatten_weight



# class WeightsLoader:
#     def __init__(self, darknet_state, tail_state):
#         self.darknet_state = darknet_state.copy()
#         self.tail_state = tail_state.copy()
#     def load_weights(self, weight_path):
#
#
# # for key in darknet.state_dict():
# #     print(key)
# #
# # for key in tail.state_dict():
# #     print(key)
#
# #print("======================================================")
darknet, tail, weight = load_checkpoint(darknet, tail, weights)
