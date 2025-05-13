import argparse, os
from ultralytics import YOLO
from threading import Thread

IMAGE_SIZE = 640

def _parse_arguments():

    parser = argparse.ArgumentParser("Create Synthetic Dataset")

    parser.add_argument(
        "-d", "--dataset",
        help="Directory of dataset to train from",
        required=True,
    )

    parser.add_argument(
        "-o", "--output",
        help="Directory where results should be save",
        required=True,
    )

    parser.add_argument(
        "-m", "--model",
        help="Model file that should be trained from (default: yolov8m.pt)",
        default="yolov8m.pt"
    )

    parser.add_argument(
        "--freeze",
        help="Freeze modules except the 9th, 8th, and 5th module",
        action="store_true"
    )

    parser.add_argument(
        "--epochs",
        help="Specify the number or epochs to train the network",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--save_period",
        help="Save every nth epoch",
        type=int, 
        default=5,
    )

    parser.add_argument(
        "--patience",
        help="Maximum patience for improvement",
        type=int,
        default=40,
    )

    args = vars(parser.parse_args())

    print(args["dataset"])

    args["dataset"] = os.path.join(args["dataset"], "data.yaml")

    print(args["dataset"])


    if not os.path.exists(args["dataset"]) :
        print(f"Dataset - {args['dataset']} - is not valid")
        exit()

    if not os.path.exists(args["output"]) :
        os.makedirs(args["output"])

    return args

def _freeze_backbone(trainer):

    model = trainer.model
    
    freeze_number = 10
    
    print("-"*40 + "Freezing Layers".center(20) + "-"*40)

    exempt = [f"model.{9}.", f"model.{8}.", f"model.{5}."]
    freeze = [f"model.{x}." for x in range(freeze_number) if f"model.{x}." not in exempt]

    for name, param in model.named_parameters():
        param.requires_grad = True

        if name[:8] in freeze:
            print(f"Freezing layer: {name}")
            param.requires_grad = False

        elif name == "model.22.dfl.conv.weight" :
            print(f"Freezing layer: {name}")
            param.requires_grad = False

def _train_model(program_arguments):

    model = YOLO(program_arguments["model"])

    if program_arguments["freeze"] :
        model.add_callback("on_train_start", _freeze_backbone)

    results = model.train(
        data=program_arguments["dataset"], 
        epochs=program_arguments["epochs"], 
        imgsz=IMAGE_SIZE, 
        save=True, 
        save_period=program_arguments["save_period"],
        project=program_arguments["output"],
        patience=program_arguments["patience"]
    )

    print(f"The final mAP score: {results.results_dict['metrics/mAP50(B)']}")

def main():

    program_arguments = _parse_arguments()

    t = Thread(target=_train_model, args=(program_arguments,))
    t.start()
    t.join()

if __name__ == "__main__" :
    main()