import os, shutil, argparse

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
        default="downsampled_dataset"
    )

    parser.add_argument(
        "-p", "--percentage",
        help="The percentage to which the dataset should be reduced to (0 - 100)",
        required=True
    )

    args = vars(parser.parse_args())
    
    args["input"] = os.path.abspath(args["input"])
    args["output"] = os.path.abspath(args["output"])
    args["percentage"] = int(args["percentage"])

    if not os.path.exists(args["input"]) :
        print(f"Input directory - {args['input']} - does not exist")
        exit()

    _create_YOLO_directory(args["output"])

    return args

def main():

    program_arguments = _parse_arguments()

    shutil.rmtree(os.path.join(program_arguments["output"], "images", "val"))
    shutil.rmtree(os.path.join(program_arguments["output"], "labels", "val"))

    shutil.copytree(os.path.join(program_arguments["input"], "images", "val"), os.path.join(program_arguments["output"], "images", "val"))
    shutil.copytree(os.path.join(program_arguments["input"], "labels", "val"), os.path.join(program_arguments["output"], "labels", "val"))

    all_train_images = os.listdir(os.path.join(program_arguments["input"], "images", "train"))
    all_train_images.sort()
   
    number_of_images = len(all_train_images)
    number_of_output_images = int( number_of_images*program_arguments["percentage"]/100 )

    for I in range(number_of_output_images):

        target_idx = int(number_of_output_images*I/number_of_output_images)

        filename = all_train_images[target_idx]
        label_name = ".".join(all_train_images[target_idx].split(".")[:-1]) + ".txt"

        shutil.copy( os.path.join(program_arguments["input"], "images", "train", filename), os.path.join(program_arguments["output"], "images", "train", filename))
        shutil.copy( os.path.join(program_arguments["input"], "labels", "train", label_name), os.path.join(program_arguments["output"], "labels", "train", label_name))

if __name__ == "__main__" :
    main()