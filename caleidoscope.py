import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torchvision
import PIL
from PIL import Image

image_path = "/home/ivan/Desktop/angle/100"
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(1, 1, 1)
image = cv2.imread(f"{image_path}.jpg")

ANGLE = 45
height, width, _ = image.shape
original_shape = image.shape
cv2.rectangle(image, (0, 0), (width - 1, height - 1), color=(255, 0, 0), thickness=10)
# plt.imshow(image)
# plt.show()

# region mirror image
center = cv2.hconcat([cv2.flip(image, 1), image, cv2.flip(image, 1)])
image = cv2.vconcat([cv2.flip(cv2.flip(center, -1), 1), center, cv2.flip(cv2.flip(center, -1), 1)])
# plt.imshow(image)
# plt.show()
# endregion

# # region rotate image
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img)

rotated_image = torchvision.transforms.functional.rotate(im_pil, ANGLE, resample=False, expand=True, fill=0)
print(rotated_image)
# plt.imshow(rotated_image)
# plt.show()

image = np.array(rotated_image.convert('RGB'))[:, :, ::-1].copy()
# region crop image
caleidoscop_height, caleidoscop_width, _ = image.shape
rotated_height, rotated_width,_ = image.shape

SCALE = rotated_height / 3*height, \
        rotated_width / 3*width

scale_h =( int(caleidoscop_height / 3) - int(2 * caleidoscop_height / 3) )/ height
scale_w =(int(caleidoscop_width / 3) - int(2 * caleidoscop_width / 3))/ width

image = image[int(caleidoscop_height/ 3):int(2*caleidoscop_height / 3),
        int(caleidoscop_width / 3):int(2*caleidoscop_width/ 3)]
# plt.imshow(image)
# plt.show()
#cv2.imwrite(image_path+"augmented.jpg", image)


import numpy as np
import uuid
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
from src.boxes_manipulations import normalize_boxes as nb, \
    denormalize_boxes as db, \
    convert_to_matplotlib as cm, convert_to_yolo as cy, current_classes_dictionary, validate_boxes
names = ["class_index", "center_x", "center_y", "w", "h"]


original_boxes = pd.read_csv(f"{image_path}.txt", header=None, sep=' ').values
original_boxes= pd.DataFrame(original_boxes, columns=names)
transfromed_boxes = pd.read_csv(f"{image_path}augmented.txt", header=None, sep=' ').values
transformed_boxes = pd.DataFrame(transfromed_boxes, columns=names)


rotation_matrix = np.array([[math.cos(ANGLE), -math.sin(ANGLE)],
                            [math.sin(ANGLE), math.cos(ANGLE)]])

transformed_boxes[["center_x", "center_y"]] = np.dot(transformed_boxes[["center_x", "center_y"]]/np.array([scale_w, scale_h]), rotation_matrix)#- np.array([(scale_w-1)/2, (scale_h-1)/2])

oB = db(original_boxes[names].values, original_shape)
tB = db(transformed_boxes[names].values, original_shape)

#transformed_boxes[[["center_x", "center_y"]] = transformed_boxes[[["center_x", "center_y"]]

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(1, 1, 1)
for class_index, l, t, w, h in cm(oB):
    ax.add_patch(plt.Rectangle((l, t), int(w), int(h), linewidth=4, color="b", fill=False))
for class_index, l, t, w, h in cm(tB):
    ax.add_patch(plt.Rectangle((l, t), int(w), int(h), linewidth=4, color="r", fill=False))

plt.imshow(cv2.imread(f"{image_path}.jpg"))
plt.show()