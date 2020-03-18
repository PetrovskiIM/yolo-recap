import cmath
from cmath import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SQRT_OF_2 = sqrt(2)


def solve(w0, h0, w1, h1):
    w, h, cosa, sina = 0, 0, 1, 0
    st = ""
    if ((w0 / h0) >= (w1 / h1)) & (w0 / h0 >= 1):
        w, h, cosa, sina = solute(w0, h0, w1, h1, 0)
        st += "done"
    elif ((w0 / h0) < (w1 / h1)) & (w0 / h0 > 1):
        st += "no cases"
        w, h,  sina, cosa = solute(h0, w0, w1, h1, 1)
    elif ((w0 / h0) > (w1 / h1)) & (w0 / h0 < 1):
        st += "the one part seems to work extreamly well"
        w, h, sina, cosa = solute(h0, w0, w1, h1, 0)
    elif ((w0 / h0) < (w1 / h1)) & (w0 / h0 < 1):
        st += "need more ifs"
        w, h, cosa, sina = solute(w0, h0, w1, h1, 1)
    return w.real, h.real, np.array([[cosa.real, -sina.real],
                                     [sina.real, cosa.real]])


def solute(alpha_rotated_container_width,
           alpha_rotated_container_height,
           sum_rotated_container_width,
           sum_rotated_container_height, axis):
    if axis == 0:
        w0, h0 = alpha_rotated_container_width, alpha_rotated_container_height
        w1, h1 = sum_rotated_container_width, sum_rotated_container_height
    else:
        h0, w0 = alpha_rotated_container_width, alpha_rotated_container_height
        h1, w1 = sum_rotated_container_width, sum_rotated_container_height

    denominator = w0 ** 2 - SQRT_OF_2 * w0 * w1 + w1 ** 2
    cosa = sqrt(2 + (sqrt(w0 ** 2 * (2 * denominator - h0 ** 2)) - (h0 * w0) + SQRT_OF_2 * h0 * w1) / denominator) / 2
    sina = sqrt(1 - cosa ** 2)
    w = (cosa + sina) * w0 - SQRT_OF_2 * sina * w1
    h = (sina - cosa) * w0 + SQRT_OF_2 * cosa * w1
    if axis == 0:
        return w, h, cosa, sina
    else:
        return h, w, cosa, sina


boxes = pd.read_csv("boxes.csv")[["center_x", "center_y", "width", "height", "45width", "45height"]]

ws = []
hs = []
rotation_matrixis = []

for i in range(len(boxes)):
    w0, h0, w1, h1 = boxes.iloc[i][["width", "height", "45width", "45height"]].values
    if (w1 is None) | (h1 is None):
        ws.append(0)
        hs.append(0)
        rotation_matrixis.append(np.array([[1, 0], [0, 1]]))
    else:
        w, h, rotation_matrix = solve(w0, h0, w1, h1)
        ws.append(w)
        hs.append(h)
        rotation_matrixis.append(rotation_matrix)
boxes["w"] = ws
boxes["h"] = hs
boxes["rotation_matrix"] = rotation_matrixis
import cv2

image_path = "/home/ivan/Desktop/angle/1003"
image = cv2.imread(f"{image_path}.jpg")
fig = plt.figure(dpi=380)
ax = fig.add_subplot(1, 1, 1)
plt.axis('off')
for i in range(len(boxes)):
    center_x, center_y, w, h, matrix = boxes.iloc[i][["center_x", "center_y", "w", "h", "rotation_matrix"]].values
    coordinates = np.dot((np.array([[0, 0],
                                    [w, 0],
                                    [w, h],
                                    [0, h],
                                    [0, 0]]) - [w / 2, h / 2]), matrix) + [center_x, center_y]
    w0, h0, w1, h1 = boxes.iloc[i][["width", "height", "45width", "45height"]].values
    ax.plot(coordinates[:, 0], coordinates[:, 1])
    try:
        if h0.real != 0:
            plt.text(coordinates[0, 0], coordinates[0, 1], str(round(w0.real / h0.real, 3)), fontsize=3, color="r")

            plt.text(coordinates[0, 0], coordinates[0, 1] + 8, str(round(w1.real / h1.real, 3)), fontsize=3, color="b")
    except:
        1 + 1
plt.imshow(image)
plt.show()