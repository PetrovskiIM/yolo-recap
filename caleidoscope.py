from src.boxes_manipulations import normalize_boxes, denormalize_boxes, convert_to_matplotlib, \
    convert_to_yolo, current_classes_dictionary, validate_boxes, convert_to_voc
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as functional
import math

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
scaled_boxes = pd.read_csv(f"{image_path}.txt", header=None, sep=' ').values

transformed_image_path = "/home/ivan/Desktop/angle/1003augmented"
transformed_image = rotate_mirroring_corners(image, 30)
cv2.imwrite(f"{transformed_image_path}.jpg", transformed_image, )

angle_in_radians = math.pi / 6
transformed_image = cv2.imread(f"{transformed_image_path}.jpg")
scaled_transfromed_boxes = pd.read_csv(f"{image_path}augmented.txt", header=None, sep=' ').values
absolute_boxes = coordinate_transform(denormalize_boxes(scaled_boxes, image.shape), angle_in_radians, image)
absolute_transformed_boxes = denormalize_boxes(scaled_transfromed_boxes, transformed_image.shape)

# region visualization
fig = plt.figure(dpi=380)
ax = fig.add_subplot(1, 1, 1)
# for _, x, y, w, h in absolute_boxes:
# if w / h > 1.8:
#     ax.plot([x], [y], "D", color="r")
# # if h /w  > 1.8:
#     ax.plot([x], [y], "D", color="r")

# if h/w>2:
#     ax.plot([x], [y], "D", color="r")

for _, x, y, w, h in absolute_transformed_boxes:
    if (h / w > 1.5):
        ax.plot([x], [y], "s", color="b")
# if h / w > 1.8:
#     ax.plot([x], [y], "s", color="b")
# if w / h > 1.8:
#     ax.plot([x], [y], "s", color="b")

plt.imshow(transformed_image)
plt.show()

# image_path = "/home/ivan/Desktop/angle/1003"
# image = cv2.imread(f"{image_path}.jpg")
# scaled_boxes = pd.read_csv(f"{image_path}.txt", header=None, sep=' ').values

# transformed_image_path = "/home/ivan/Desktop/angle/1003augmented"
# transformed_image = rotate_mirroring_corners(image, 30)
# cv2.imwrite(f"{transformed_image_path}.jpg", transformed_image, )

angle_in_radians = -math.pi / 6
# transformed_image = cv2.imread(f"{transformed_image_path}.jpg")
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
alphta_boxes =pd.DataFrame(absolute_transformed_boxes, columns=["class_index", "center_x", "center_y", "width", "height"]).astype(float)

correspondence = []
for box in boxes.to_dict("records"):
    sizes_of_similar = []
    for alpha_box in alphta_boxes.loc[alphta_boxes["center_x"]>0].loc[alphta_boxes["center_y"]>0].to_dict("records"):
        if np.sqrt(np.dot(
                np.array([box["center_x"],box["center_y"]]),
                np.array([alpha_box["center_x"], alpha_box["center_y"]]))
        ) < 10:
            sizes_of_similar.append({ "width":alpha_box["width"], "height": alpha_box["height"]})
    correspondence.append(sizes_of_similar)
classes = []
boxes["correspondence"] = correspondence
for box in boxes.to_dict("records"):
    if len(box["correspondence"]):
        if (box["width"]>box["height"]) & box["correspondence"]["width"]/box["correspondence"]["height"]
    else:
        classes.append("?")


