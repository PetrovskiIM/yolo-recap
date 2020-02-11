import pandas as pd

current_classes_dictionary = {"nroi": 99, "roi": 100, "pricep": 7}


def area(box_dictionary):
    return (box_dictionary["xmax"] - box_dictionary["xmin"]) * (box_dictionary["ymax"] - box_dictionary["ymin"])


def ratio(box_dictionary):
    return (box_dictionary["xmax"] - box_dictionary["xmin"]) / (box_dictionary["ymax"] - box_dictionary["ymin"])


def convert_to_yolo(scaled_voc_boxes, one_class=True):
    scaled_voc_boxes = filter_classes(scaled_voc_boxes, ["roi", "nroi", "pricep"], current_classes_dictionary)
    ab_txt_lines = []
    for box in scaled_voc_boxes:
        class_index, xmi, ymi, xma, yma = box
        if one_class:
            class_index = 0
        width, height = xma - xmi, yma - ymi
        ab_txt_lines.append([int(class_index), xmi + width / 2, ymi + height / 2, width, height])
    return ab_txt_lines


def normalize_boxes(voc_boxes, shape):
    image_height, image_width, _ = shape
    if len(voc_boxes) > 0:
        boxes = pd.DataFrame(voc_boxes, columns=["class_index", "xmin", "ymin", "xmax", "ymax"]).astype(float)
        boxes[["xmin", "xmax"]] = boxes[["xmin", "xmax"]] / image_width
        boxes[["ymin", "ymax"]] = boxes[["ymin", "ymax"]] / image_height
        return boxes[["class_index", "xmin", "ymin", "xmax", "ymax"]].values
    else:
        return []


def denormalize_boxes(ab_scaled_boxes, shape):
    image_height, image_width, _ = shape
    if len(ab_scaled_boxes) > 0:
        scaled_boxes = pd.DataFrame(ab_scaled_boxes,
                                    columns=["class_index", "center_x", "center_y", "width", "height"]).astype(float)
        scaled_boxes["xmin"] = (scaled_boxes["center_x"] - scaled_boxes["width"] / 2) * image_width
        scaled_boxes["xmax"] = (scaled_boxes["center_x"] + scaled_boxes["width"] / 2) * image_width
        scaled_boxes["ymin"] = (scaled_boxes["center_y"] - scaled_boxes["height"] / 2) * image_height
        scaled_boxes["ymax"] = (scaled_boxes["center_y"] + scaled_boxes["height"] / 2) * image_height
        return scaled_boxes[["class_index", "xmin", "ymin", "xmax", "ymax"]].values
    else:
        return []


def validate_boxes(boxes, shape):
    image_height, image_width, _ = shape
    if len(boxes) > 0:
        boxes = pd.DataFrame(boxes, columns=["class_index", "xmin", "ymin", "xmax", "ymax"]).round().astype(int)
        boxes.loc[boxes["xmin"] < 0, "xmin"] = 0
        boxes.loc[boxes["ymin"] < 0, "ymin"] = 0
        boxes.loc[boxes["xmax"] > image_width, "xmax"] = int(image_width)
        boxes.loc[boxes["ymax"] > image_height, "ymax"] = int(image_height)
        boxes["area"] = (boxes["xmax"] - boxes["xmin"]) * (boxes["ymax"] - boxes["ymin"])
        return boxes.loc[boxes["area"] > 1, ["class_index", "xmin", "ymin", "xmax", "ymax"]].round().astype(int).values
    else:
        return []


def convert_to_matplotlib(voc_boxes):
    matplotlib_boxes = []
    for class_index, xmin, ymin, xmax, ymax in voc_boxes:
        matplotlib_boxes.append([class_index, xmin, ymin, xmax - xmin, ymax - ymin])
    return matplotlib_boxes


def convert_to_coco(voc_boxes):
    coco_boxes = []
    categories_ids = []
    for box in voc_boxes:
        class_index, xmi, ymi, xma, yma = box
        width, height = xma - xmi, yma - ymi
        left, top = xmi, ymi
        coco_boxes.append([int(left), int(top), int(width), int(height)])
        categories_ids.append(class_index)
    return categories_ids, coco_boxes


def filter_classes(boxes, unwelcome_classes, classes_dictionary=current_classes_dictionary):
    boxes_frame = pd.DataFrame(boxes, columns=["class_index", "xmin", "ymin", "xmax", "ymax"])
    if len(unwelcome_classes):
        indicator_of_welcome_classes = \
            (boxes_frame["class_index"].astype(int) != classes_dictionary[unwelcome_classes[0]])
        for unwelcome_class in unwelcome_classes:
            indicator_of_welcome_classes = \
                indicator_of_welcome_classes & (boxes_frame["class_index"].astype(int) != classes_dictionary[
                    unwelcome_class])
        return boxes_frame.loc[indicator_of_welcome_classes][["class_index", "xmin", "ymin", "xmax", "ymax"]].values
    return boxes


def convert_to_voc(yolo_boxes):
    vox_boxes = []
    for class_index, center_x, center_y, width, height in yolo_boxes:
        xmin, ymin, xmax, ymax = center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2
        vox_boxes.append([class_index, xmin, ymin, xmax, ymax])
    return vox_boxes
