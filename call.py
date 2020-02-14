from src.boxes_manipulations import normalize_boxes, denormalize_boxes, convert_to_matplotlib, \
    convert_to_yolo, current_classes_dictionary, validate_boxes, convert_to_voc
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as functional
import math
import requests
import json
import base64
from scipy.ndimage.interpolation import rotate


def convert_cv2_to_pil(cv2_image):
    from PIL import Image
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def convert_pil_to_cv2(pil_image):
    import cv2
    import numpy as np
    return np.array(pil_image.convert('RGB'))[:, :, ::-1].copy()


def rotate_mirroring_corners(image, angle):
    height, width, _ = image.shape
    cv2.rectangle(image, (0, 0), (width, height), color=(255, 0, 0), thickness=3)
    center_line = cv2.hconcat([cv2.flip(image, 1), image, cv2.flip(image, 1)])
    mirrored_image = cv2.vconcat(
        [cv2.flip(cv2.flip(center_line, -1), 1), center_line, cv2.flip(cv2.flip(center_line, -1), 1)])
    rotated_mirrored_image = convert_pil_to_cv2(
        functional.rotate(convert_cv2_to_pil(mirrored_image),
                          angle=angle, center=(image.shape[1] * 3 / 2, image.shape[0] * 3 / 2),
                          resample=False, expand=True, fill=0))
    height, width, _ = rotated_mirrored_image.shape
    print(f"mirrored {width / 3} {height / 3}")
    return rotated_mirrored_image[int(height / 3): int(2 * height / 3), int(width / 3): int(2 * width / 3)]


names = ["class_index", "center_x", "center_y", "width", "height"]

image_path = "/home/ivan/Desktop/angle/1003"
image = cv2.imread(f"{image_path}.jpg")
boxes = pd.read_csv(f"{image_path}.txt", header=None, sep=' ').values

transformed_image_path = "/home/ivan/Desktop/angle/1003augmented"
# transformed_image = rotate_mirroring_corners(image, 30)
# cv2.imwrite(f"{transformed_image_path}.jpg", transformed_image,)
transformed_image = cv2.imread(f"{transformed_image_path}.jpg")
transfromed_boxes = pd.read_csv(f"{image_path}augmented.txt", header=None, sep=' ').values

print(f"transformed shape: {transformed_image.shape}, original shape: {image.shape}")

absolute_boxes = denormalize_boxes(boxes, image.shape)
p_angle =angle = math.pi / 6

scale = math.sin(angle) + math.cos(angle)
height, width, _ = image.shape
transformed_height, transformed_width, _ = transformed_image.shape
new_center = np.array((0 + math.sin(angle)/2 + math.cos(angle)/ 2, math.cos(angle) / 2 + math.sin(angle) / 2))
rotation_matrix = np.array([[math.cos(p_angle), -math.sin(p_angle)],
                            [math.sin(p_angle), math.cos(p_angle)]])
if len(absolute_boxes):
    boxes_ex = pd.DataFrame(boxes,
                            columns=["class_index", "center_x", "center_y", "width", "height"]).astype(float)
    # region flip -> move to center -> rotate -> move to center -> denormilize
    boxes_ex["center_y"] = boxes_ex["center_y"].values
    boxes_ex["center_x"] = boxes_ex["center_x"].values
    boxes_ex[["center_x", "center_y"]] = boxes_ex[["center_x", "center_y"]].values + [-1 / 2, -1 / 2]
    moved = boxes_ex[["center_x", "center_y"]].values
    boxes_ex[["center_x", "center_y"]] = np.matmul(boxes_ex[["center_x", "center_y"]].values, rotation_matrix)
    rot = boxes_ex[["center_x", "center_y"]].values
    boxes_ex[["center_x", "center_y"]] = boxes_ex[["center_x", "center_y"]].values + new_center
    new_c = boxes_ex[["center_x", "center_y"]].values
    # endregion


# region visualization
fig = plt.figure(dpi=380)
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.array([0, math.cos(angle), math.cos(angle) + math.sin(angle), math.sin(angle), 0]),
        np.array([math.sin(angle), 0, math.cos(angle), math.cos(angle) + math.sin(angle), math.sin(angle)]), color="g", linewidth=2)

xs = np.array([0, 1, 1, 0, 0])
ys = np.array([0, 0, 1, 1, 0])

# box = np.array([xs,ys])
# ax.plot(xs,ys, color="b")
#
# for _, x, y, _, _ in boxes:
#     ax.plot([x], [y], 'b+')
#
# ax.plot(xs - 1/2, ys - 1/2, color="yellow")
# for x, y in moved:
#     ax.plot([x], [y], 'yo')
#
# moved_box = np.array([xs -1/2,ys-1/2])
# ax.plot(np.matmul(moved_box.T, rotation_matrix)[:,0], np.matmul(moved_box.T,rotation_matrix)[:,1], color="black")
# for x, y in rot:
#     ax.plot([x], [y], 'bo', color="black")
# ax.plot(np.matmul(moved_box.T, rotation_matrix)[:,0] + scale / 2, np.matmul(moved_box.T, rotation_matrix)[:, 1] + scale / 2, color="pink")

for x, y in new_c:
    ax.plot([x], [y], 'g+')
ax.plot([0, scale, scale, 0,      0],
        [0, 0,     scale,  scale, 0], color="red")
for _, x, y, _, _ in transfromed_boxes:
    ax.plot([x*scale], [y*scale], 'r+', color="r")
plt.show()
# endregion

