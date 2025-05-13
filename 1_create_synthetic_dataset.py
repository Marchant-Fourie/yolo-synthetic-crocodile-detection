import argparse, os, json, shutil, random
import numpy as np
import cv2

from _utils import (
    _load_json_file,
    _box_around_polygon,
    _generate_YOLO_annotations,
    _box_within_box,
    _create_YOLO_directory,
    _save_text_file,
    _save_image,
)

SCALES = [0.66666, 1.0, 1.33333]
TARGET_WIDTH = 640
TARGET_HEIGHT = 640

def _parse_arguments() -> dict:

    parser = argparse.ArgumentParser("Create Synthetic Dataset")

    parser.add_argument(
        "-i", "--input",
        help="Directory of raw synthetic data (default: raw_synthetic_data)",
        default="raw_synthetic_data"
    )

    parser.add_argument(
        "-o", "--output",
        help="Directory where synthetic dataset should be stored (default: synthetic_dataset)",
        default="synthetic_dataset"
    )

    parser.add_argument(
        "-n", "--negative",
        help="Whether to include images that do not have crocodiles",
        action="store_true"
    )

    args = vars(parser.parse_args())
    
    args["input"] = os.path.abspath(args["input"])
    args["output"] = os.path.abspath(args["output"])

    if not os.path.exists(args["input"]) :
        print(f"Input directory - {args['input']} - does not exist")
        exit()

    _create_YOLO_directory(args["output"])

    return args

def _sample_from_annotation(image: np.ndarray, annotations: dict, image_filename: str, idx : int, scale: int) -> tuple:
    
    image_data: dict = annotations[image_filename]
    polygons = image_data["polygons"]

    image, polygons = _scale_image_and_polygons(image, polygons, scale)

    height = image.shape[0]
    width = image.shape[1]

    target_polygon = polygons[idx]

    left, top, right, bottom = _box_around_polygon(target_polygon)

    crop_left = random.randint(max(right - TARGET_WIDTH, 0), min(left, width - TARGET_WIDTH ))
    crop_top = random.randint(max(bottom - TARGET_HEIGHT, 0), min(top, height - TARGET_HEIGHT ))
    crop_right = crop_left + TARGET_WIDTH
    crop_bottom = crop_top + TARGET_HEIGHT

    cropped_image = image[crop_top:crop_bottom,crop_left:crop_right,:]

    all_boxes = []

    for poly in polygons:

        box = _box_around_polygon(poly)
        contained_box = _box_within_box(box, [crop_left, crop_top, crop_right, crop_bottom])

        if contained_box != None:
            all_boxes.append(contained_box)

    image_annotations = _generate_YOLO_annotations(all_boxes, [TARGET_WIDTH, TARGET_HEIGHT])

    return cropped_image, image_annotations

def _negative_sample_from_annotation(image: np.ndarray, annotations: dict, image_filename: str, idx : int, scale: int, num_attempts = 100) -> tuple:
    
    image_data: dict = annotations[image_filename]
    polygons = image_data["polygons"]

    image, polygons = _scale_image_and_polygons(image, polygons, scale)

    height = image.shape[0]
    width = image.shape[1]

    for I in range(num_attempts):

        crop_left = random.randint(0, width - TARGET_WIDTH)
        crop_top = random.randint(0, height - TARGET_HEIGHT)
        crop_right = crop_left + TARGET_WIDTH
        crop_bottom = crop_top + TARGET_HEIGHT

        all_boxes = []

        for poly in polygons:

            box = _box_around_polygon(poly)
            contained_box = _box_within_box(box, [crop_left, crop_top, crop_right, crop_bottom])

            if contained_box != None:
                all_boxes.append(contained_box)

        if len(all_boxes) == 0:
            
            cropped_image = image[crop_top:crop_bottom,crop_left:crop_right,:]
            image_annotations = ""

            return cropped_image, image_annotations
    
    return None, None

def _scale_image_and_polygons(image: np.ndarray, polygons: list, scale: float) -> tuple:

    height = image.shape[0]
    width = image.shape[1]

    new_width = int(width*scale)
    new_height = int(height*scale)

    image = cv2.resize(image, (new_width, new_height))

    output_polygons = []

    for polygon in polygons:

        x_points = []
        y_points = []

        for x, y in zip(polygon[0], polygon[1]):

            x_points.append( int(round(x*scale)) )
            y_points.append( int(round(y*scale)) )

        output_polygons.append([x_points, y_points])

    return image, output_polygons

# def draw_annotations_on_image(annotations, image):

#     for line in annotations.split("\n"):
        
#         if line.replace(" ", "") == "" :
#             continue

#         _, center_x, center_y, width, height = line.split(" ")

#         center_x = float(center_x)*TARGET_WIDTH
#         center_y = float(center_y)*TARGET_HEIGHT
#         width = float(width)*TARGET_WIDTH
#         height = float(height)*TARGET_HEIGHT

#         left = np.clip(int(center_x - width/2), 0, TARGET_WIDTH - 1)
#         right = np.clip(int(center_x + width/2), 0, TARGET_WIDTH - 1)
#         top = np.clip(int(center_y - height/2), 0, TARGET_HEIGHT - 1)
#         bottom = np.clip(int(center_y + height/2), 0, TARGET_HEIGHT - 1)

#         image = draw_rect([left, top, right, bottom], image)

#     return image

# def draw_rect(box, image):

#     left, top, right, bottom = box

#     for x in range(left, right):
        
#         image[top, x, 2] = 255
#         image[top, x, 1] = 0
#         image[top, x, 0] = 0

#         image[bottom, x, 2] = 255
#         image[bottom, x, 1] = 0
#         image[bottom, x, 0] = 0

#     for y in range(top, bottom):
        
#         image[y, left, 2] = 255
#         image[y, left, 1] = 0
#         image[y, left, 0] = 0

#         image[y, right, 2] = 255
#         image[y, right, 1] = 0
#         image[y, right, 0] = 0

#     return image

def main():

    program_arguments = _parse_arguments()

    input_dir = program_arguments["input"]
    output_dir = program_arguments["output"]

    annotations = _load_json_file(os.path.join(input_dir, "annotations.json"))

    all_images = list(annotations.keys())

    counter = 0 

    for image_filename in all_images:

        if not os.path.exists(os.path.join(input_dir, image_filename)) :
            print(f"Image - {os.path.join(input_dir, image_filename)} - was not found")
            continue

        print(f"Processing - {os.path.join(input_dir, image_filename)}")

        img = cv2.imread(os.path.join(input_dir, image_filename))

        for scale in SCALES:

            for idx in range(len(annotations[image_filename]["polygons"])) :
                
                save_filename = f"{image_filename.replace('.png', '')}_{counter}"
                counter += 1

                cropped_image, image_annotations = _sample_from_annotation(np.copy(img), annotations, image_filename, idx, scale)

                _save_image(os.path.join(output_dir, "images", "train", save_filename + ".png"), cropped_image)
                _save_text_file(os.path.join(output_dir, "labels", "train", save_filename + ".txt"), image_annotations)

            if program_arguments["negative"]:

                for idx in range(len(annotations[image_filename]["polygons"])) :
                    
                    save_filename = f"{image_filename.replace('.png', '')}_{counter}"
                    counter += 1

                    cropped_image, image_annotations = _negative_sample_from_annotation(np.copy(img), annotations, image_filename, idx, scale)

                    if image_annotations == None:
                        print(f"Could not find negative image")
                        continue

                    _save_image(os.path.join(output_dir, "images", "train", save_filename + ".png"), cropped_image)
                    _save_text_file(os.path.join(output_dir, "labels", "train", save_filename + ".txt"), image_annotations)




if __name__ == "__main__" :
    main()
