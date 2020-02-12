from src.boxes_manipulations import normalize_boxes, \
    denormalize_boxes, \
    convert_to_matplotlib, \
    convert_to_yolo, \
    current_classes_dictionary, \
    validate_boxes, \
    convert_to_voc
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math
import torchvision.transforms.functional as functional


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
    cv2.rectangle(image, (0, 0), (width, height), color=(255, 0, 0), thickness=10)
    center_line = cv2.hconcat([cv2.flip(image, 1), image, cv2.flip(image, 1)])
    mirrored_image = cv2.vconcat(
        [cv2.flip(cv2.flip(center_line, -1), 1), center_line, cv2.flip(cv2.flip(center_line, -1), 1)])
    # region visualization
    fig = plt.figure(dpi=380)
    ax = fig.add_subplot(1, 1, 1)
    #transformed_image_1 = cv2.imread(f"{transformed_image_path}.jpg")
    plt.imshow(mirrored_image)
    plt.show()
    # endregion
    print(height)
    print(width)
    rotated_mirrored_image = convert_pil_to_cv2(
        functional.rotate(convert_cv2_to_pil(mirrored_image),
                          angle,
                        #  center=(width/2, height/2),
                          resample=False,
                          expand=True,
                          fill=0))
    height, width, _ = rotated_mirrored_image.shape
    return rotated_mirrored_image[int(height / 3): int(2 * height / 3), int(width / 3): int(2 *width / 3)]


def coordinate_transform(scaled_yolo_boxes, transformed_shape, angle, shape):
    yolo_boxes = denormalize_boxes(scaled_yolo_boxes, transformed_shape)
    height, width, _ = shape
    transformed_height, transformed_width, _ = transformed_shape
    len_of_transformed_diagonal = np.sqrt(np.dot(transformed_shape, transformed_shape))
    rotation_matrix = np.array([[math.cos(angle), math.sin(angle)],
                                [-math.sin(angle), math.cos(angle)]])
    if len(yolo_boxes) > 0:
        boxes = pd.DataFrame(yolo_boxes, columns=["class_index", "center_x", "center_y", "width", "height"]).astype(
            float)

        # region visialization
        fig = plt.figure(dpi=380)
        ax = fig.add_subplot(1, 1, 1)
        transformed_image_1 = cv2.imread(f"{transformed_image_path}.jpg")
        for _, l, t, w, h in convert_to_matplotlib(
                convert_to_voc(boxes[["class_index", "center_x", "center_y", "width", "height"]].values)):
            ax.plot([l+w/2], [t+h/2], 'r+', color="r")  # ,scalex=False, scaley=False,
            ax.add_patch(plt.Rectangle((l, t), int(w), int(h), linewidth=2, color="b", fill=False))
        plt.imshow(transformed_image_1)
        plt.show()
        # endregion

        boxes[["center_x", "center_y"]] = np.dot(
            boxes[["center_x", "center_y"]].values - [transformed_width / 2, transformed_height / 2], rotation_matrix) \
                                          + [len_of_transformed_diagonal / 2, len_of_transformed_diagonal / 2]

        # region visualization
        fig = plt.figure(dpi=380)
        ax = fig.add_subplot(1, 1, 1)
        for x,y in boxes[["center_x", "center_y"]].values:
            ax.plot([x],[y], 'r+',color="r")#,scalex=False, scaley=False,
            #print(f"{x},{y}")
        rotated_back = convert_pil_to_cv2(functional.rotate(convert_cv2_to_pil(transformed_image_1),
                                                            -45,
                                                            resample=False,
                                                            expand=True,
                                                            fill=0))
        cv2.rectangle(rotated_back,
                      (int((len_of_transformed_diagonal - width) / 2),
                       int((len_of_transformed_diagonal - height) / 2)),
                      (int((len_of_transformed_diagonal - width) / 2 + width),
                       int((len_of_transformed_diagonal - height) / 2 + height)), color=(0, 255, 0), thickness=2)
        plt.imshow(rotated_back)
        plt.savefig('./filename.png', dpi=300)
        plt.show(style="empty")
        # endregion
    else:
        return []

    return normalize_boxes(boxes[["class_index", "center_x", "center_y", "width", "height"]].values, shape)


names = ["class_index", "center_x", "center_y", "width", "height"]

image_path = "/home/ivan/Desktop/angle/100"
image = cv2.imread(f"{image_path}.jpg")
# image = cv2.resize(image, (600,400))

transformed_image_path = "/home/ivan/Desktop/angle/100augmented"
transformed_image = rotate_mirroring_corners(image, 45)

# region visualization
fig = plt.figure(dpi=380)
ax = fig.add_subplot(1, 1, 1)
# transformed_image_1 = cv2.imread(f"{transformed_image_path}.jpg")
plt.imshow(transformed_image)
plt.show()
# endregion
cv2.imwrite(f"{transformed_image_path}.jpg", transformed_image)

#transformed_image = cv2.imread(f"{transformed_image_path}.jpg")

print(f"transformed shape: {transformed_image.shape}, original shape: {image.shape}")
boxes = pd.read_csv(f"{image_path}.txt", header=None, sep=' ').values

transfromed_boxes = pd.read_csv(f"{image_path}augmented.txt", header=None, sep=' ').values

transfromed_boxes_in_original_axis = coordinate_transform(transfromed_boxes, transformed_image.shape, 315, image.shape)

fig = plt.figure(dpi=380)
ax = fig.add_subplot(1, 1, 1)
for class_index, l, t, w, h in convert_to_matplotlib(denormalize_boxes(convert_to_voc(boxes), image.shape)):
    ax.add_patch(plt.Rectangle((l, t), int(w), int(h), linewidth=1, color="b", fill=False))
# for class_index, l, t, w, h in convert_to_matplotlib(
#         denormalize_boxes(convert_to_voc(transfromed_boxes_in_original_axis), image.shape)):
#     ax.add_patch(plt.Rectangle((l, t), int(w), int(h), linewidth=1, color="r", fill=False))

plt.imshow(cv2.imread(f"{image_path}.jpg"))
plt.show()
