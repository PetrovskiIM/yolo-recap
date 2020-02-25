from src.boxes_manipulations import denormalize_boxes
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as functional
import math
import numpy as np

names = ["class_index", "center_x", "center_y", "width", "height"]


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


def coordinate_transform(boxes, angle, image):
    height, width, _ = image.shape
    expandex_height, expandex_width = math.sin(angle) * width + math.cos(angle) * height, \
                                      math.sin(angle) * height + math.cos(angle) * width
    rotated_mirroring_corners_image_center = np.array([expandex_width, expandex_height]) / 2
    rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                [math.sin(angle), math.cos(angle)]])
    if len(boxes):
        boxes = pd.DataFrame(boxes, columns=["class_index", "center_x", "center_y", "width", "height"]).astype(float)
        boxes[["center_x", "center_y"]] = (np.matmul(boxes[["center_x", "center_y"]].values + [-width / 2, -height / 2],
                                                     rotation_matrix) + rotated_mirroring_corners_image_center)
        return boxes[["class_index", "center_x", "center_y", "width", "height"]].values
    else:
        return []


def coordinates_reverse_transform(boxes, angle, image, transformed_image):
    height, width, _ = transformed_image.shape
    shrink_height, shrink_width, _ = image.shape
    original_image_center = np.array([shrink_width, shrink_height]) / 2
    rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                [math.sin(angle), math.cos(angle)]])
    if len(boxes):
        boxes = pd.DataFrame(boxes, columns=["class_index", "center_x", "center_y", "width", "height"]).astype(float)
        boxes[["center_x", "center_y"]] = (np.matmul(boxes[["center_x", "center_y"]].values + [-width / 2, -height / 2],
                                                     rotation_matrix) + original_image_center)
        return boxes[["class_index", "center_x", "center_y", "width", "height"]].values
    else:
        return []


image_path = "/home/ivan/Desktop/angle/1003"
image = cv2.imread(f"{image_path}.jpg")


transformed_image_path = "/home/ivan/Desktop/angle/1003augmented"
transformed_image = rotate_mirroring_corners(image, 45)
cv2.imwrite(f"{transformed_image_path}.jpg", transformed_image, )
angle_in_radians = math.pi / 4
# transformed_image = cv2.imread(f"{transformed_image_path}.jpg")

scaled_boxes = pd.read_csv(f"{image_path}.txt", header=None, sep=' ').values
scaled_transfromed_boxes = pd.read_csv(f"{image_path}augmented.txt", header=None, sep=' ').values
absolute_boxes = coordinate_transform(denormalize_boxes(scaled_boxes, image.shape), angle_in_radians, image)
absolute_transformed_boxes = denormalize_boxes(scaled_transfromed_boxes, transformed_image.shape)


# region visualization
fig = plt.figure(dpi=380)
ax = fig.add_subplot(1, 1, 1)
for _, x, y, w, h in absolute_boxes:
    beta = math.pi/4
    martrix = np.array([[math.cos(beta), -math.sin(beta)],[math.sin(beta), math.cos(beta)]])
    coordinates = np.dot(np.array([[0, 0], [w, 0], [w, h], [0, h], [0, 0]]) - [w / 2, h / 2],martrix) + [x, y]
    ax.plot(coordinates[:, 0], coordinates[:, 1], color="r")

for _, x, y, w, h in absolute_transformed_boxes:
    coordinates = np.array([[0, 0], [w, 0], [w, h], [0, h], [0, 0]]) - [w / 2, h / 2] + [x, y]
    ax.plot(coordinates[:, 0], coordinates[:, 1], color="b")
plt.imshow(transformed_image)
plt.show()

angle_in_radians = -math.pi / 4
scaled_transfromed_boxes = pd.read_csv(f"{image_path}augmented.txt", header=None, sep=' ').values
absolute_transformed_boxes = coordinates_reverse_transform(
    denormalize_boxes(scaled_transfromed_boxes, transformed_image.shape),
    angle_in_radians, image, transformed_image)
absolute_boxes = denormalize_boxes(scaled_boxes, image.shape)

# region visualization
fig = plt.figure(dpi=380)
ax = fig.add_subplot(1, 1, 1)
for _, x, y, w, h in absolute_boxes:
    ax.plot([x], [y], "r+")
for _, x, y, w, h in absolute_transformed_boxes:
    ax.plot([x], [y], "r+", color="b")
plt.imshow(image)
plt.show()

boxes = pd.DataFrame(absolute_boxes, columns=["class_index", "center_x", "center_y", "width", "height"]).astype(float)
alphta_boxes = pd.DataFrame(absolute_transformed_boxes,
                            columns=["class_index", "center_x", "center_y", "width", "height"]).astype(float)

paired_boxes = []
for box in boxes.to_dict("records"):
    taked_in_accout = False
    box = box.copy()
    box["45width"] = None
    box["45height"] = None
    for alpha_box in alphta_boxes.to_dict("records"):
        if np.sqrt(np.dot(
                np.array([box["center_x"], box["center_y"]]) - np.array([alpha_box["center_x"], alpha_box["center_y"]]),
                np.array([box["center_x"], box["center_y"]]) - np.array([alpha_box["center_x"], alpha_box["center_y"]]))
        ) < 10:
            box["45width"] = alpha_box["width"]  #  image.shape[1] / transformed_image.shape[1] * alpha_box["width"]
            box["45height"] = alpha_box["height"] #  image.shape[0] / transformed_image.shape[0] * alpha_box["height"]
            paired_boxes.append(box)
            taked_in_accout = True
    if not taked_in_accout:
        paired_boxes.append(box)

pd.DataFrame.from_records(paired_boxes, index=None).to_csv("boxes.csv")
# boxes.loc[boxes["width"] > boxes["height"]]
# for box in boxes.to_dict("records"):
#     if len(box["correspondence"]):
#         if (box["width"]>box["height"]) & box["correspondence"]["width"]/box["correspondence"]["height"]
#     else:
#         classes.append("?")
