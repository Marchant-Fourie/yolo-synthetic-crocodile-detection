import os, shutil, argparse, cv2
import numpy as np
from _utils import _create_YOLO_directory

def _parse_arguments():

    parser = argparse.ArgumentParser("Create Synthetic Dataset")

    parser.add_argument(
        "-i", "--input",
        help="Directory of dataset to sample from (default: real_world_dataset)",
        default="synthetic_dataset"
    )

    parser.add_argument(
        "-o", "--output",
        help="Directory where dataset should be stored",
        default="augmented_dataset"
    )

    parser.add_argument(
        "-g", "--gaussian",
        help="The sigma of the gaussian blurring that should be added",
        default=0,
        type=float
    )

    parser.add_argument(
        "-n", "--noise",
        help="The sigma of the gaussian noise that should be added",
        default=0,
        type=float
    )

    args = vars(parser.parse_args())
    
    args["input"] = os.path.abspath(args["input"])
    args["output"] = os.path.abspath(args["output"])
    args["gaussian"] = float(args["gaussian"])
    args["noise"] = float(args["noise"])

    if args["gaussian"] == 0 and args["noise"] == 0:
        print("No gaussian or noise value was specified")
        exit()

    if not os.path.exists(args["input"]) :
        print(f"Input directory - {args['input']} - does not exist")
        exit()

    _create_YOLO_directory(args["output"])

    return args

def _add_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    
    kernel_size = max(3, int(6*sigma + 1))

    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)

def _add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:   
    
    height = img.shape[0]
    width = img.shape[1]

    img = img.astype(np.float32)
    img = img + np.random.normal(0, sigma, (height, width, 3))

    img = np.clip(img, 0, 255)
    img = np.round(img)
    img = img.astype(np.uint8)

    return img

def main():

    program_arguments = _parse_arguments()

    shutil.rmtree(os.path.join(program_arguments["output"], "images", "val"))
    shutil.rmtree(os.path.join(program_arguments["output"], "labels", "val"))
    shutil.rmtree(os.path.join(program_arguments["output"], "labels", "train"))

    shutil.copytree(os.path.join(program_arguments["input"], "images", "val"), os.path.join(program_arguments["output"], "images", "val"))
    shutil.copytree(os.path.join(program_arguments["input"], "labels", "val"), os.path.join(program_arguments["output"], "labels", "val"))
    shutil.copytree(os.path.join(program_arguments["input"], "labels", "train"), os.path.join(program_arguments["output"], "labels", "train"))

    train_path = os.path.join(program_arguments["input"], "images", "train")
    output_train_path = os.path.join(program_arguments["output"], "images", "train")

    all_train_images = os.listdir(train_path)
    all_train_images.sort()
   
    for image_file in all_train_images:

        print(f"Processing {image_file}")

        img = cv2.imread(os.path.join(train_path, image_file), cv2.IMREAD_UNCHANGED)

        if program_arguments["gaussian"] != 0:
            img = _add_gaussian_blur(img, program_arguments["gaussian"])

        if program_arguments["noise"] != 0:
            img = _add_gaussian_noise(img, program_arguments["noise"])

        cv2.imwrite(os.path.join(output_train_path, image_file), img)

if __name__ == "__main__" :
    main()