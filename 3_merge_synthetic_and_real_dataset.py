import os, shutil, argparse

from _utils import _create_YOLO_directory

def _parse_arguments():

    parser = argparse.ArgumentParser("Create Synthetic Dataset")

    parser.add_argument(
        "-a", "--a",
        help="Directory of dataset A to merge from",
        required=True,
    )

    parser.add_argument(
        "-b", "--b",
        help="Directory of dataset B to merge from",
        required=True,
    )

    parser.add_argument(
        "-o", "--output",
        help="Directory where dataset should be stored",
        required=True,
    )

    args = vars(parser.parse_args())
    
    args["a"] = os.path.abspath(args["a"])
    args["b"] = os.path.abspath(args["b"])
    args["output"] = os.path.abspath(args["output"])

    _create_YOLO_directory(args["output"])

    return args

def _copy_directory_with_extension(source: str, target: str, suffix: str):

    source_files = os.listdir(source)

    for file in source_files:
        shutil.copy(os.path.join(source, file), os.path.join(target, suffix + "_" + file))

def main():

    program_arguments = _parse_arguments()

    _copy_directory_with_extension(os.path.join(program_arguments["a"], "images", "train"), os.path.join(program_arguments["output"], "images", "train"), "a")
    _copy_directory_with_extension(os.path.join(program_arguments["a"], "labels", "train"), os.path.join(program_arguments["output"], "labels", "train"), "a")
    _copy_directory_with_extension(os.path.join(program_arguments["a"], "images", "val"), os.path.join(program_arguments["output"], "images", "val"), "a")
    _copy_directory_with_extension(os.path.join(program_arguments["a"], "labels", "val"), os.path.join(program_arguments["output"], "labels", "val"), "a")

    _copy_directory_with_extension(os.path.join(program_arguments["b"], "images", "train"), os.path.join(program_arguments["output"], "images", "train"), "b")
    _copy_directory_with_extension(os.path.join(program_arguments["b"], "labels", "train"), os.path.join(program_arguments["output"], "labels", "train"), "b")
    _copy_directory_with_extension(os.path.join(program_arguments["b"], "images", "val"), os.path.join(program_arguments["output"], "images", "val"), "b")
    _copy_directory_with_extension(os.path.join(program_arguments["b"], "labels", "val"), os.path.join(program_arguments["output"], "labels", "val"), "b")

if __name__ == "__main__" :
    main()