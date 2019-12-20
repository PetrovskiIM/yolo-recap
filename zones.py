import numpy as np
import random
import pandas as pd
from torch import Tensor, cat, sigmoid, exp, meshgrid, linspace, stack


def estimate_distance(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    x_min, y_min, x_max, y_max = min(x1_min, x2_min), min(y1_min, y2_min), max(x1_max, x2_max), max(y1_max, y2_max)
    width, height = x_max - x_min, y_max - y_min
    width1, height1 = x1_max - x1_min, y1_max - y1_min
    width2, height2 = x2_max - x2_min, y2_max - y2_min
    return width - (width1 + width2), height - (height1 + height2)


def estimate_luft(origin_box, boxes, roi):
    x_min_roi, y_min_roi, x_max_roi, y_max_roi = roi
    xes, yes = [x_min_roi, x_max_roi], [y_min_roi, y_max_roi]
    for x_min, y_min, x_max, y_max in boxes:
        xes.append(x_min)
        xes.append(x_max)
        yes.append(y_min)
        yes.append(y_max)
    for x_min in xes:
        for x_max in xes:
            if x_max > x_min:
                boxes.append([x_min, roi[0] - 1, x_max, roi[0] - 1])
                boxes.append([x_min, roi[3], x_max, roi[3] + 1])
    for y_min in yes:
        for y_max in yes:
            if y_max > x_min:
                boxes.append([roi[0] - 1, y_min, roi[0], y_max])
                boxes.append([roi[2], y_min, roi[2] + 1, y_max])
    boxes.append([roi[0] - 1, roi[1] - 1, roi[0], roi[1]])
    boxes.append([roi[0] - 1, roi[3] - 1, roi[0], roi[3]])
    boxes.append([roi[2], roi[3], roi[0] + 1, roi[3] + 1])
    boxes.append([roi[2], roi[3], roi[2] + 1, roi[3] + 1])

    areas = []
    for i in range(len(boxes)):
        valid = True
        area = found_pack(origin_box, luft_box(origin_box, boxes[i]))
        # print(area)
        for j in range(len(boxes)):
            if (i != j) & is_intersected(boxes[j], area):
                valid = False
        if valid:
            areas.append(area)
    return list(areas)


def is_intersected(box1, box2):
    return (estimate_distance(box1, box2)[0] < 0) & (estimate_distance(box1, box2)[1] < 0)


def luft_box(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return max(min(x1_max, x2_min), min(x1_min, x2_max)), max(min(y1_min, y2_max), min(y1_max, y2_min)), min(
        max(x1_min, x2_max), max(x2_min, x1_max)), min(max(y1_min, y2_max),
                                                       max(y2_min, y1_max))  # true = max max in the


def found_pack(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return min(x1_min, x2_min), min(y1_min, y2_min), max(x1_max, x2_max), max(y1_max, y2_max)


def form_packs(boxes):
    is_boxes_disjoint = False
    while not is_boxes_disjoint:
        is_boxes_disjoint = True

        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if (i != j) & is_intersected(boxes[i], boxes[j]):
                    is_boxes_disjoint = False
                    break
        origin_box = boxes[0]
        fused_boxes = boxes[1:]
        survaivers = []
        for j, box in enumerate(fused_boxes):
            if is_intersected(origin_box, box):
                origin_box = found_pack(origin_box, box)
                is_boxes_disjoint = False
            else:
                survaivers.append(box)
        survaivers.append(origin_box)
        boxes = survaivers.copy()
    return boxes


def ramp_up_box(box, luft, roi):
    return [max(0, box[0] - luft[0]),
            max(0, box[1] - luft[1]),
            min(roi[2], box[2] + luft[2]),
            min(roi[3], box[3] + luft[3])]


def form_save_packs(boxes, luft, roi):
    is_boxes_disjoint = False
    while not is_boxes_disjoint:
        is_boxes_disjoint = True
        origin_box = boxes[0]
        fused_boxes = boxes[1:]
        survaivers = []
        for j, box in enumerate(fused_boxes):
            if is_intersected(ramp_up_box(origin_box, luft, roi), ramp_up_box(box, luft, roi)):
                origin_box = found_pack(origin_box, box)
                is_boxes_disjoint = False
            else:
                survaivers.append(box)
        survaivers.append(origin_box)
        boxes = survaivers.copy()

        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if (i != j) & is_intersected(ramp_up_box(boxes[i], luft, roi), ramp_up_box(boxes[j], luft, roi)):
                    is_boxes_disjoint = False
                    break

    return boxes


def is_edge_leakers(box, padding, roi):
    return (np.abs(box[0] - roi[0]) < padding) | (np.abs(box[1] - roi[1]) < padding) | (
            np.abs(box[2] - roi[2]) < padding) | (np.abs(box[3] - roi[3]) < padding)


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def available_growth(pack, packs, roi, edge):
    vertical_packs = packs.loc[((pack["xmin"] < packs["xmax"]) & (pack["xmax"] >= packs["xmax"]))
                               | ((pack["xmin"] <= packs["xmin"]) & (pack["xmax"] > packs["xmin"]))
                               | ((pack["xmin"] >= packs["xmin"]) & (pack["xmax"] <= packs["xmax"]))]

    horizontal_packs = packs.loc[((pack["ymin"] < packs["ymax"]) & (pack["ymax"] >= packs["ymax"]))
                                 | ((pack["ymin"] <= packs["ymin"]) & (pack["ymax"] > packs["ymin"]))
                                 | ((pack["ymin"] >= packs["ymin"]) & (pack["ymax"] <= packs["ymax"]))]

    up = vertical_packs.loc[vertical_packs["ymin"] >= pack["ymax"]]
    down = vertical_packs.loc[vertical_packs["ymax"] <= pack["ymin"]]
    left = horizontal_packs.loc[horizontal_packs["xmax"] <= pack["xmin"]]
    right = horizontal_packs.loc[horizontal_packs["xmin"] >= pack["xmin"]]

    if len(left) > 0:
        left_record = max(left["xmax"])
    else:
        for i in range(len(roi) - 1):
            x = 0
            if intersection(line((edge[0], pack["ymin"]), (pack["xmin"], pack["ymin"])),
                            line(roi.iloc[i].values, roi.iloc[i + 1].values)):
                x_down, y_down = intersection(line((edge[0], pack["ymin"]), (pack["xmin"], pack["ymin"])),
                                              line(roi.iloc[i].values, roi.iloc[i + 1].values))
                if (x_down >= x) & (x_down <= pack["xmin"]):
                    x = x_down
            if intersection(line((edge[0], pack["ymax"]), (pack["xmin"], pack["ymax"])),
                            line(roi.iloc[i].values, roi.iloc[i + 1].values)):
                x_up, y_up = intersection(line((edge[0], pack["ymax"]), (pack["xmin"], pack["ymax"])),
                                          line(roi.iloc[i].values, roi.iloc[i + 1].values))
                if (x_up > x) & (x_up <= pack["xmin"]):
                    x = x_up
        left_record = x
    if len(right) > 0:
        right_record = min(right["xmin"])
    else:
        x = edge[2]
        for i in range(len(roi) - 1):
            if intersection(line((edge[2], pack["ymin"]), (pack["xmax"], pack["ymin"])),
                            line(roi.iloc[i].values, roi.iloc[i + 1].values)):
                x_down, y_down = intersection(line((edge[2], pack["ymin"]), (pack["xmax"], pack["ymin"])),
                                              line(roi.iloc[i].values, roi.iloc[i + 1].values))
                if (x_down <= x) & (x_down >= pack["xmax"]):
                    x = x_down
            if intersection(line((edge[2], pack["ymax"]), (pack["xmax"], pack["ymax"])),
                            line(roi.iloc[i].values, roi.iloc[i + 1].values)):
                x_up, y_up = intersection(line((edge[2], pack["ymax"]), (pack["xmax"], pack["ymax"])),
                                          line(roi.iloc[i].values, roi.iloc[i + 1].values))
                if (x_up <= x) & (x_up >= pack["xmax"]):
                    x = x_up
        right_record = x
    if len(up) > 0:
        up_record = min(up["ymin"])
    else:
        y = edge[3]
        for i in range(len(roi) - 1):
            if intersection(line((pack["xmin"], edge[3]), (pack["xmin"], pack["ymax"])),
                            line(roi.iloc[i].values, roi.iloc[i + 1].values)):
                x_down, y_down = intersection(line((pack["xmin"], edge[1]), (pack["xmin"], pack["ymax"])),
                                              line(roi.iloc[i].values, roi.iloc[i + 1].values))
                if (y_down < y) & (y_down >= pack["ymax"]):
                    y = y_down
            if intersection(line((pack["xmax"], edge[3]), (pack["xmax"], pack["ymax"])),
                            line(roi.iloc[i].values, roi.iloc[i + 1].values)):
                x_up, y_up = intersection(line((pack["xmax"], edge[1]), (pack["xmax"], pack["ymax"])),
                                          line(roi.iloc[i].values, roi.iloc[i + 1].values))
                if (y_up < y) & (y_up >= pack["ymax"]):
                    y = y_up
        up_record = y
    if len(down) > 0:
        down_record = max(down["ymax"])
    else:
        y = edge[1]
        for i in range(len(roi) - 1):
            if intersection(line((pack["xmin"], edge[3]), (pack["xmin"], pack["ymax"])),
                            line(roi.iloc[i].values, roi.iloc[i + 1].values)):
                x_down, y_down = intersection(line((pack["xmin"], edge[1]), (pack["xmin"], pack["ymax"])),
                                              line(roi.iloc[i].values, roi.iloc[i + 1].values))
                if (y_down >= y) & (y_down <= pack["ymin"]):
                    y = y_down
            if intersection(line((pack["xmax"], edge[3]), (pack["xmax"], pack["ymax"])),
                            line(roi.iloc[i].values, roi.iloc[i + 1].values)):
                x_up, y_up = intersection(line((pack["xmax"], edge[1]), (pack["xmax"], pack["ymax"])),
                                          line(roi.iloc[i].values, roi.iloc[i + 1].values))
                if (y_up >= y) & (y_up <= pack["ymin"]):
                    y = y_up
        down_record = y
    return left_record, up_record, right_record, down_record


def choose_square(save_zone, pack):
    luft = [save_zone["xmin"] - pack["xmin"],
            save_zone["ymin"] - pack["ymin"],
            pack["xmax"] - save_zone["xmax"],
            pack["ymax"] - save_zone["ymax"]]
    luft = [int(l) for l in luft]

    width = int(save_zone["xmax"] - save_zone["xmin"])
    height = int(save_zone["ymax"] - save_zone["ymin"])

    available_width = (luft[0] + width + luft[2])
    available_height = (luft[1] + height + luft[3])
    available_square_side = min(available_width, available_height)

    if max(width, height) > available_square_side:
        # print("pointless")
        if available_width > available_height:
            # changed until nice version come out
            # return [int(save_zone["xmin"]-min(width/2, luft[0])),
            #         int(pack["ymin"]),
            #         int(save_zone["xmax"]+min(width/2, luft[2])),
            #         int(pack["ymax"])]
            return []
        else:
            return []
            # return [int(pack["xmin"]), int(save_zone["ymin"] - min(luft[1], 100)),
            #         int(pack["xmax"]), int(save_zone["ymax"] + min(luft[3], 100))]

    width_delta = available_width - available_square_side
    height_delta = available_height - available_square_side

    if width_delta == 0:
        left_luft = luft[0]
        right_luft = luft[2]
    else:
        if (available_square_side - width) - luft[0] < 0:
            if luft[2] == 0:
                right_luft = 0
                left_luft = available_square_side - width
            else:
                right_luft = random.randint(0, min(luft[2], (available_square_side - width)))
                left_luft = available_square_side - width - right_luft
        elif (available_square_side - width) - luft[2] < 0:
            if luft[0] == 0:
                left_luft = 0
                right_luft = available_square_side - width
            else:
                right_luft = random.randint(0, min(luft[2], (available_square_side - width)))
                left_luft = available_square_side - width - right_luft
            left_luft = random.randint(0, min(luft[0], (available_square_side - width)))
            right_luft = available_square_side - width - left_luft
        else:
            free_luft = luft[0] + width + luft[2] - available_square_side
            if luft[0] > luft[2]:
                right_luft = min(luft[2], available_square_side - width)
                left_luft = available_square_side - width - right_luft
            else:
                left_luft = min(luft[0], available_square_side - width)
                right_luft = available_square_side - width - left_luft

    if height_delta == 0:
        up_luft = luft[1]
        down_luft = luft[3]
    else:
        if (available_square_side - height) - luft[1] < 0:
            if luft[3] == 0:
                down_luft = 0
                up_luft = available_square_side - height
            else:
                down_luft = random.randint(0, min(luft[3], (available_square_side - height)))
                up_luft = available_square_side - height - down_luft
        elif (available_square_side - height) - luft[3] < 0:
            if luft[1] == 0:
                up_luft = 0
                down_luft = available_square_side - height
            else:
                up_luft = random.randint(0, min(luft[1], (available_square_side - height)))
                down_luft = available_square_side - height - up_luft
        else:
            free_luft = luft[1] + height + luft[3] - available_square_side
            if luft[1] > luft[3]:
                down_luft = min(luft[3], available_square_side - height)
                up_luft = available_square_side - height - down_luft
            else:
                up_luft = min(luft[1], available_square_side - height)
                down_luft = available_square_side - height - up_luft
    square = [int(save_zone["xmin"] - left_luft),
              int(save_zone["ymin"] - up_luft),
              int(save_zone["xmax"] + right_luft),
              int(save_zone["ymax"] + down_luft)]
    return square


def soft_squares(image, boxes):
    image_height, image_width = image.shape[:2]
    edge = [0, 0, image_width, image_height]
    dots = [
        {
            "x": image_width,
            "y": 0
        },
        {
            "x": 0,
            "y": 0
        },
        {
            "x": 0,
            "y": image_height
        },
        {
            "x": image_width,
            "y": image_height
        },
        {
            "x": image_width,
            "y": 0
        },
        {
            "x": 0,
            "y": 0
        }
    ]
    roi = pd.DataFrame.from_records(dots)
    squares = []
    box_packs = pd.DataFrame(form_packs(boxes), columns=["xmin", "ymin", "xmax", "ymax"])
    for i, pack in box_packs.iterrows():
        save_zone = pack.copy()
        packs = box_packs.loc[box_packs.index != i].copy()

        grouth = available_growth(pack, packs, roi, edge)
        while np.abs(pack["ymin"] - grouth[3]) + np.abs(pack["ymax"] - grouth[1]) + \
                np.abs(grouth[0] - pack["xmin"]) + np.abs(pack["xmax"] - grouth[2]) > 0.2:
            width = pack["xmax"] - pack["xmin"]
            height = pack["ymax"] - pack["ymin"]
            if width < height:
                if max(np.abs(grouth[0] - pack["xmin"]), np.abs(pack["xmax"] - grouth[2])) < 0.00001:
                    if np.abs(pack["ymin"] - grouth[3]) > np.abs(pack["ymax"] - grouth[1]):
                        pack["ymin"] = grouth[3]
                    else:
                        pack["ymax"] = grouth[1]

                else:
                    if np.abs(grouth[0] - pack["xmin"]) > np.abs(pack["xmax"] - grouth[2]):
                        pack["xmin"] = grouth[0]
                    else:
                        pack["xmax"] = grouth[2]
            else:
                if max(np.abs(grouth[0] - pack["xmin"]), np.abs(pack["xmax"] - grouth[2])) < 0.00001:
                    if np.abs(pack["ymin"] - grouth[3]) > np.abs(pack["ymax"] - grouth[1]):
                        pack["ymin"] = grouth[3]
                    else:
                        pack["ymax"] = grouth[1]
                else:
                    if np.abs(grouth[0] - pack["xmin"]) > np.abs(pack["xmax"] - grouth[2]):
                        pack["xmin"] = grouth[0]
                    else:
                        pack["xmax"] = grouth[2]
            grouth = available_growth(pack, packs, roi, edge)

        crop_square = choose_square(save_zone, pack)
        if len(crop_square) != 0:
            if crop_square[2] - crop_square[0] > 200:
                squares.append(choose_square(save_zone, pack))
    return squares


def soft_save_squares(image, boxes, luft):
    image_height, image_width = image.shape[:2]
    edge = [0, 0, image_width, image_height]
    dots = [
        {
            "x": image_width,
            "y": 0
        },
        {
            "x": 0,
            "y": 0
        },
        {
            "x": 0,
            "y": image_height
        },
        {
            "x": image_width,
            "y": image_height
        },
        {
            "x": image_width,
            "y": 0
        },
        {
            "x": 0,
            "y": 0
        }
    ]
    roi = pd.DataFrame.from_records(dots)
    squares = []
    box_packs = pd.DataFrame(form_save_packs(boxes, luft, [0, 0, image_width, image_height]),
                             columns=["xmin", "ymin", "xmax", "ymax"])
    for i, pack in box_packs.iterrows():
        save_zone = pack.copy()
        packs = box_packs.loc[box_packs.index != i].copy()

        grouth = available_growth(pack, packs, roi, edge)
        while np.abs(pack["ymin"] - grouth[3]) + np.abs(pack["ymax"] - grouth[1]) + \
                np.abs(grouth[0] - pack["xmin"]) + np.abs(pack["xmax"] - grouth[2]) > 0.2:
            width = pack["xmax"] - pack["xmin"]
            height = pack["ymax"] - pack["ymin"]
            if width < height:
                if max(np.abs(grouth[0] - pack["xmin"]), np.abs(pack["xmax"] - grouth[2])) < 0.00001:
                    if np.abs(pack["ymin"] - grouth[3]) > np.abs(pack["ymax"] - grouth[1]):
                        pack["ymin"] = grouth[3]
                    else:
                        pack["ymax"] = grouth[1]

                else:
                    if np.abs(grouth[0] - pack["xmin"]) > np.abs(pack["xmax"] - grouth[2]):
                        pack["xmin"] = grouth[0]
                    else:
                        pack["xmax"] = grouth[2]
            else:
                if max(np.abs(grouth[0] - pack["xmin"]), np.abs(pack["xmax"] - grouth[2])) < 0.00001:
                    if np.abs(pack["ymin"] - grouth[3]) > np.abs(pack["ymax"] - grouth[1]):
                        pack["ymin"] = grouth[3]
                    else:
                        pack["ymax"] = grouth[1]
                else:
                    if np.abs(grouth[0] - pack["xmin"]) > np.abs(pack["xmax"] - grouth[2]):
                        pack["xmin"] = grouth[0]
                    else:
                        pack["xmax"] = grouth[2]
            grouth = available_growth(pack, packs, roi, edge)

        crop_square = choose_square(save_zone, pack)
        if len(crop_square) != 0:
            if crop_square[2] - crop_square[0] > 200:
                squares.append(choose_square(save_zone, pack))
    return squares
