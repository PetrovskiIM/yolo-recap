import numpy as np
import pandas as pd
import matplotlib
import cv2
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import torch
from torch import Tensor, cat, sigmoid, exp, stack
from model import Darknet, Tail, Head
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

path = "/home/ivan/sets/up/e000009f0000000000000005/Dolgoprudny/Russia/Prospekt-Raketostroiteley--5k1/20190831154940"


def get_image(path):
    if os.path.isfile(path):
        if path.endswith("png"):
            return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        elif path.endswith("jpg") | path.endswith("JPEG") | path.endswith("jpeg"):
            return cv2.imread(path)
    else:
        return None


def get_bboxes(path, shape):
    image_height, image_width, _ = shape
    ab_txt_lines = []
    if os.path.isfile(path):
        try:
            ab_txt_lines = pd.read_csv(path, header=None, sep=' ').values
        except pd.errors.EmptyDataError:
            ab_txt_lines = []
    return [{
        "xmin": (scaled_center_x - scaled_width / 2) * image_width,
        "xmax": (scaled_center_x + scaled_width / 2) * image_width,
        "ymin": (scaled_center_y - scaled_height / 2) * image_height,
        "ymax": (scaled_center_y + scaled_height / 2) * image_height
    } for class_index, scaled_center_x, scaled_center_y, scaled_width, scaled_height in ab_txt_lines]


def convert_to_matplotlib(voc_boxes):
    matplotlib_boxes = []
    for xmin, ymin, xmax, ymax in voc_boxes:
        matplotlib_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
    return matplotlib_boxes


def convert_to_yolo(scaled_voc_boxes):
    ab_txt_lines = []
    for box in scaled_voc_boxes:
        class_index, xmi, ymi, xma, yma = box
        width, height = xma - xmi, yma - ymi
        ab_txt_lines.append([int(class_index), xmi + width / 2, ymi + height / 2, width, height])
    return ab_txt_lines


image = get_image(path + ".jpg")
boxes = pd.DataFrame.from_records(get_bboxes(path + ".txt", image.shape))[["xmin", "ymin", "xmax", "ymax"]].values
fig = plt.figure(figsize=(24, 64))
ax1 = fig.add_subplot(1, 1, 1)
# for left, bottom, w, h in convert_to_matplotlib(boxes):
#     p = plt.Rectangle((left, bottom), w, h, color="r", linewidth=1, fill=False)
#     ax1.add_patch(p)
# ax1.imshow(image)

# from zones import form_packs, soft_squares

packs = []  # form_packs(boxes)
zones = []  # soft_squares(image, packs)

ax1.imshow(image)

boxes = pd.DataFrame(boxes, columns=["x_min", "y_min", "x_max", "y_max"])
xs = list(boxes["x_min"].values)
ys = list(boxes["y_min"].values)
xs.extend(boxes["x_max"].values)
ys.extend(boxes["y_max"].values)
xs.append(0)
xs.append(1920)
ys.append(0)
ys.append((1080))

lower_area_threshold = 0.1
upper_area_threshold = 1
max_number_of_zone = 100
min_zones_intersection = 0.3
xs = list(set(xs))
xs.sort()
ys = list(set(ys))
ys.sort()
zones = []
I = 0
is_x_min_compromised = False
is_y_min_compromised = False
is_x_max_compromised = False
is_y_max_compromised = False
for x_min in xs:
    for y_min in ys:
        if is_x_min_compromised:
            is_x_min_compromised = False
            break
        for x_max in xs:
            if is_x_min_compromised | is_y_min_compromised:
                is_x_min_compromised = False
                is_y_min_compromised = False
                break
            if x_max < x_min + 300:
                continue
            for y_max in ys:
                if is_x_min_compromised | is_y_min_compromised | is_x_max_compromised:
                    is_x_min_compromised = False
                    is_y_min_compromised = False
                    is_x_max_compromised = False
                    break
                if y_max < y_min + 300:
                    continue
                print(I)
                I += 1
                boxes_involved_by_y = boxes.loc[((boxes["y_min"] > y_min) & (boxes["y_min"] < y_max)) |
                                                ((boxes["y_max"] > y_min) & (boxes["y_max"] < y_max))]

                is_x_min_compromised = len(boxes_involved_by_y.loc[(boxes_involved_by_y["x_min"] < x_min) & (
                        boxes_involved_by_y["x_max"] > x_min)]) > 0
                if is_x_min_compromised:
                    break
                is_x_max_compromised = len(boxes_involved_by_y.loc[(boxes_involved_by_y["x_max"] > x_max) & (
                        boxes_involved_by_y["x_min"] < x_max)]) > 0
                if is_x_max_compromised:
                    break
                boxes_involved_by_x = boxes.loc[((boxes["x_min"] > x_min) & (boxes["x_min"] < x_max)) |
                                                ((boxes["x_max"] > y_min) & (boxes["x_max"] < y_max))]
                is_y_min_compromised = len(boxes_involved_by_x.loc[(boxes_involved_by_x["y_min"] < y_min) & (
                            boxes_involved_by_x["y_max"] > y_min)]) > 0
                if is_y_min_compromised:
                    break
                is_y_max_compromised = len(boxes_involved_by_x.loc[(boxes_involved_by_x["y_max"] > y_max) & (
                            boxes_involved_by_x["y_min"] < y_max)]) > 0
                if is_y_max_compromised:
                    continue
                zones.append([x_min, y_min, x_max, y_max])

fig = plt.figure(figsize=(24, 64))
ax1 = fig.add_subplot(1, 1, 1)
for left, bottom, w, h in convert_to_matplotlib(boxes):
    p = plt.Rectangle((left, bottom), w, h, color="#ffccdf", linewidth=2, fill=False)
    ax1.add_patch(p)
for left, bottom, w, h in convert_to_matplotlib(zones):
    p = plt.Rectangle((left, bottom), w, h, color="#9bf2ef", linewidth=2, fill=False)
    ax1.add_patch(p)
ax1.imshow(image)
