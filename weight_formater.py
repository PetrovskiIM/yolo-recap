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



#
#
#
# class Weights:
#     def __init__(self, darknet_weight_file_path):
#         with open(darknet_weight_file_path, "rb") as f:
#             self.header = np.fromfile(f, dtype=np.int32, count=5)
#             self.seen = self.header[3]
#             self.flatten_weights = np.fromfile(f, dtype=np.float32)
#
#     def match_keys_for_layer_with_normalization_for_layer_with_normalization(word, repetition_number=None, local_i=0,
#                                                                              type_number=None):
#         version_ended_by_dot = ""
#         if repetition_number is not None:
#             version_ended_by_dot += str(repetition_number) + "."
#         if type_number is not None:
#             version_ended_by_dot += str(type_number) + "."
#         return {
#             "convolution_biases": f"{word}.{version_ended_by_dot}{local_i}.bias",
#             "convolution_weights": f"{word}.{version_ended_by_dot}{local_i}.weight",
#             "weights": f"{word}.{version_ended_by_dot}{local_i + 1}.weight",
#             "biases": f"{word}.{version_ended_by_dot}{local_i + 1}.bias",
#             "running_mean": f"{word}.{version_ended_by_dot}{local_i + 1}.running_mean",
#             "running_var": f"{word}.{version_ended_by_dot}{local_i + 1}.running_var"
#         }
#
#     def match_keys_for_layer_with_bias(word, repetition_number):
#         return {
#             "convolution_biases": f"preludes.{repetition_number}.bias",
#             "convolution_weights": f"preludes.{repetition_number}.weight"
#         }
#
#     def get_state(self, model):
#         darknet, tail = model
#         def allocate_weight(key, state_dict):
#             size_expected_by_model = state_dict[key].size()
#             length_expected_by_model = np.prod(size_expected_by_model)
#             state_dict[key] = torch.from_numpy(self.flatten_weights[:length_expected_by_model]).view(size_expected_by_model)
#             self.flatten_weights = self.flatten_weights[length_expected_by_model:]
#
#         def allocate_weight_layer(darknet_to_pytorch, checkpoint, with_normalization=True):
#             if with_normalization:
#                 allocate_weight(darknet_to_pytorch["biases"], checkpoint)
#                 allocate_weight(darknet_to_pytorch["weights"], checkpoint)
#                 allocate_weight(darknet_to_pytorch["running_mean"], checkpoint)
#                 allocate_weight(darknet_to_pytorch["running_var"], checkpoint)
#             else:
#                 allocate_weight(darknet_to_pytorch["convolution_biases"], checkpoint)
#             allocate_weight(darknet_to_pytorch["convolution_weights"], checkpoint)
#
#         checkpoint = darknet.state_dict().copy()
#         number_of_yolo_layers = 3
#         allocate_weight_layer(self.match_keys_for_layer_with_normalization("intro"), checkpoint)
#         for repetition_number, num_of_types in enumerate([1, 2, 8, 8, 4]):
#             for type_number in range(num_of_types + 1):
#                 if type_number == 0:
#                     allocate_weight_layer(self.match_keys_for_layer_with_normalization("module_list", repetition_number, type_number=type_number), checkpoint)
#                 else:
#                     local_i = 0
#                     allocate_weight_layer(self.match_keys_for_layer_with_normalization("module_list", repetition_number, local_i, type_number), checkpoint)
#                     local_i += 3
#                     allocate_weight_layer(self.match_keys_for_layer_with_normalization("module_list", repetition_number, local_i, type_number), checkpoint)
#         darknet.load_state_dict(checkpoint)
#         checkpoint = tail.state_dict().copy()
#         for repetition_number in range(number_of_yolo_layers):
#             local_i = 0
#             allocate_weight_layer(self.match_keys_for_layer_with_normalization("harmonics", repetition_number, local_i),checkpoint)
#             local_i += 3
#             allocate_weight_layer(self.match_keys_for_layer_with_normalization("harmonics", repetition_number, local_i), checkpoint)
#             local_i += 3
#             allocate_weight_layer(self.match_keys_for_layer_with_normalization("harmonics", repetition_number, local_i), checkpoint)
#             local_i += 3
#             allocate_weight_layer(self.match_keys_for_layer_with_normalization("harmonics", repetition_number, local_i), checkpoint)
#             allocate_weight_layer(self.match_keys_for_layer_with_normalization("splitted_harmonic", repetition_number, type_number=0), checkpoint)
#             allocate_weight_layer(self.match_keys_for_layer_with_normalization("splitted_harmonic", repetition_number, type_number=1), checkpoint)
#             allocate_weight_layer(self.match_keys_for_layer_with_bias("preludes",repetition_number), checkpoint, with_normalization=False)
#             if repetition_number < 2:
#                     allocate_weight_layer(self.match_keys_for_layer_with_normalization("equalizers_for_routes", repetition_number), checkpoint)
#         tail.load_state_dict(checkpoint)
#         return darknet, tail
#
# weight = Weights(weight_file_path)
# darknet, tail = weight.get_state(darknet, tail)
