import os, shutil, json, cv2
import numpy as np

def _load_json_file(filename: str) -> str:

    if not os.path.exists(filename):
        print(f"Annotation file - {filename} - does not exist")
        exit()

    with open(filename, "rb") as json_file:
        output = json.loads(json_file.read())
        json_file.close()

    return output   

def _save_text_file(filename: str, content: str) -> None:

    with open(filename, "w") as f:
        f.write(content)
        f.close()

def _save_image(filename: str, image: np.ndarray):
    
    saved = cv2.imwrite(filename, image)

    if not saved:
        print(f"Failed to save - {filename}")

def _box_around_polygon(polygon: list):

    x_points = polygon[0]
    y_points = polygon[1]

    left = x_points[0]
    right = x_points[0]
    top = y_points[0]
    bottom = y_points[0]

    for x, y in zip(x_points, y_points):
        
        if x < left :
            left = x
        elif x > right:
            right = x

        if y < top :
            top = y
        elif y > bottom :
            bottom = y

    return [left, top, right, bottom]

def _box_within_box(box, larger_box):

    l_left, l_top, l_right, l_bottom = larger_box
    left, top, right, bottom = box

    width = l_right - l_left
    height = l_bottom - l_top

    left = left - l_left
    right = right - l_left
    top = top - l_top
    bottom = bottom - l_top

    if left < 0 and right < 0:
        return None
    
    if left > width and right > width: 
        return None

    if top < 0 and bottom < 0:
        return None
    
    if top > height and bottom > height: 
        return None

    o_left = max(left, 0)
    o_right = min(right, width)
    o_top = max(top, 0)
    o_bottom = min(bottom, height)

    return [o_left, o_top, o_right, o_bottom]

def _generate_YOLO_annotations(boxes: list, size: list) -> str:

    annotations = ""

    WIDTH, HEIGHT = size

    for box in boxes:

        left, top, right, bottom = box

        width = right - left
        height = bottom - top
        center_x = (right + left)/2
        center_y = (bottom + top)/2

        annotations += f"0 {center_x/WIDTH} {center_y/HEIGHT} {width/WIDTH} {height/HEIGHT}\n"

    return annotations

def _create_YOLO_directory(directory: str):

    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.makedirs(directory)

    os.makedirs(os.path.join(directory, "images", "train"))
    os.makedirs(os.path.join(directory, "images", "val"))
    os.makedirs(os.path.join(directory, "labels", "train"))
    os.makedirs(os.path.join(directory, "labels", "val"))

    data_yaml = f"path: {os.path.abspath(directory)}\ntrain: images/train\nval: images/val\n\nnames:\n  0: crocodile\n"

    with open(os.path.join(directory, "data.yaml"), "w") as f:
        f.write(data_yaml)
        f.close()

