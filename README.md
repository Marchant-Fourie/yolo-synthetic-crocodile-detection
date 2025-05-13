# Automated Crocodile Detection using Synthetic Data

This repository contains the code and scripts used to generate results for the paper titled **"Automated Crocodile Detection using Synthetic Data"**. The project leverages the Ultralytics YOLO framework for training object detection models, using both synthetic and real datasets.

## Project Structure

- `1_create_synthetic_dataset.py`:  Script to unpack and reformat the synthetic dataset into a structure compatible with Ultralytics YOLOv8.
- `2_downsample_dataset.py`: Tool for downsampling dataset into a reduced dataset.
- `3_merge_synthetic_and_real_dataset.py`: Tool for combining the real-world and synthetic datasets into a single training dataset.
- `4_augment_dataset.py`: Tool for copying and augmenting a dataset
- `5_train_YOLO_network.py`: Script to initialize training of a YOLO model on a dataset.

## Requirements

Scripts were tested on python 3.11.4, but should work on python 3.8 and upwards. To install the dependencies:

```bash
pip install -r requirements.txt
```

**Note:** For linux, you might have to use `python3` and `pip3` instead of `python` and `pip`.

## Usage

For all the scripts, the argument `--help` can be provided to get a printout of all the arguments.

### 1. Prepare the dataset

Unzip the synthetic dataset from [10.25403/UPresearchdata.28391708](10.25403/UPresearchdata.28391708) into the `raw_synthetic_data` folder and process the synthetic data into a YOLO format with:

```bash
python 1_create_synthetic_dataset.py --input raw_synthetic_data --output synthetic_data --negative
```

This process uses random values, thus the generated dataset will be different each run. The additional tag `--negative` can be removed if negative images which contain no targets should not be generated from the synthetic data. The output image size can be controlled by altering the global variables `TARGET_WIDTH` and `TARGET_HEIGHT`. The images are also captured at different scales, by multiplying the width and height with the provided scaling factors in the global variable `SCALES`.

**Note:** Unfortunately, the real-world data can not be provided due to privacy, but the scripts should still work with the provided data in the folders, as long as it follows the YOLO dataset formatting guidelines from this [website](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format).

The folder structure should be as follow:

```bash
dataset/
├── images/
│   ├── train/
│   ├── val/
├── labels/
│   ├── train/
│   ├── val/
├── data.yaml
```

Where the data.yaml has the following content:

```yaml
path: <absolute_path_to_folder/dataset>
train: images/train
val: images/val

names:
  0: crocodile
```

An example folder structure is shown in directory `real_world_dataset`, where real-world data can be placed.

### 2. Downsample the dataset

To downsample the dataset down to a specified percentage of the original datasets, the following script can be used:

```bash
python 2_downsample_dataset.py --input <path/to/dataset> --output <path/to/downsampled_dataset> --percentage <0-100>
```

### 3. Combine two datasets

To merge two datasets, such as the real-world dataset and the synthetic dataset, the following can be used:

```bash
python 3_merge_synthetic_and_real_dataset.py -a <path/to/dataset1> -b <path/to/dataset2> --output <path/to/merged_dataset>
```

### 4. Augment the dataset

Add Gaussian blurring and Gaussian noise to the images in a dataset with:

```bash
python 4_augment_dataset.py --input <path/to/dataset> --output <path/to/augmented_dataset> --blurring <sigma> --noise <sigma>
```

For the `blurring` and `noise` tag, the sigma value for the Gaussian distibution of the blurring kernel and noise distribution is provided. 

### 5. Train YOLO model on dataset

To train the model on a dataset, the `5_train_YOLO_network.py` script can be used.

The script's arguments are:

- `--dataset <path>` : Path to the dataset that should be trained on
- `--output <path>` : Path to the directory where the results should be saved
- `--model <path>` : Model that should be trained **OR** path to an already trained weights file (.pt) to use as a starting point (finetuning)
- `--freeze` : Provide tag to freeze the backbone layers except for the 9th, 8th and 5th module
- `--epochs <int>` : The number of epochs the model should train (default: 500)
- `--save_period <int>` : How many epochs should be between model saves (default: 5)
- `--patience <int>` : Maximum patience trainer waits for improvement before stopping the training early (default: 40)

Models that can be trained are the provided models from [Ultralytics](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes). 

- `yolov8n.pt`
- `yolov8s.pt`
- `yolov8m.pt`
- `yolov8l.pt`
- `yolov8x.pt`
