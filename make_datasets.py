from pathlib import Path
import argparse
import glob
import shutil
import random

"""
Make a local dataset from source images

"""


def _get_file_count(images_path):
    print(f"{images_path}/*)")
    return len(glob.glob(f"{images_path}/*"))


def _create_dataset(train_size, val_size, test_size, deer_path, background_path, model_datasets_dir):
    # 1: Create local dataset directory
    # 2: Create local train/validation/test directories
    # 3: Create deer, background directories in each of the train/val/test directories
    Path.mkdir(Path(f"{model_datasets_dir}/train/deer"), parents=True, exist_ok=True)
    Path.mkdir(Path(f"{model_datasets_dir}/train/background"), parents=True, exist_ok=True)
    Path.mkdir(Path(f"{model_datasets_dir}/validation/deer"), parents=True, exist_ok=True)
    Path.mkdir(Path(f"{model_datasets_dir}/validation/background"), parents=True, exist_ok=True)
    Path.mkdir(Path(f"{model_datasets_dir}/test/deer"), parents=True, exist_ok=True)
    Path.mkdir(Path(f"{model_datasets_dir}/test/background"), parents=True, exist_ok=True)

    # get list of all deer files
    deer_files = [f for f in Path(deer_path).glob('*')]
    random.shuffle(deer_files)
    train_deer_files = deer_files[:train_size]
    val_deer_files = deer_files[train_size:train_size + val_size]
    test_deer_files = deer_files[train_size + val_size:train_size + val_size + test_size]

    # copy the deer files
    for deer_file in train_deer_files:
        shutil.copy(deer_file, f"{model_datasets_dir}/train/deer/{deer_file.parts[-1]}")
    for deer_file in val_deer_files:
        shutil.copy(deer_file, f"{model_datasets_dir}/validation/deer/{deer_file.parts[-1]}")
    for deer_file in test_deer_files:
        shutil.copy(deer_file, f"{model_datasets_dir}/test/deer/{deer_file.parts[-1]}")

    # get list of all background files
    background_files = [f for f in Path(background_path).glob('*')]
    random.shuffle(background_files)
    train_bkgd_files = background_files[:train_size]
    val_bkgd_files = background_files[train_size:train_size + val_size]
    test_bkgd_files = background_files[train_size + val_size:train_size + val_size + test_size]

    # copy the deer files
    for bkgd_file in train_bkgd_files:
        shutil.copy(bkgd_file, f"{model_datasets_dir}/train/background/{bkgd_file.parts[-1]}")
    for bkgd_file in val_bkgd_files:
        shutil.copy(bkgd_file, f"{model_datasets_dir}/validation/background/{bkgd_file.parts[-1]}")
    for bkgd_file in test_bkgd_files:
        shutil.copy(bkgd_file, f"{model_datasets_dir}/test/background/{bkgd_file.parts[-1]}")


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deer-images", type=str, required=False, default="./datasets/kaggle/deer",
                    help="path to deer images")
    ap.add_argument("--background-images", type=str, required=False, default="./datasets/kaggle/landscape",
                    help="path to background/landscape images")
    ap.add_argument("--train-val-test", type=str, required=False, default="70,20,10",
                    help="Percentage split between train, validation and test ")
    ap.add_argument("--model-datasets-dir", type=str, required=False, default="./datasets/model",
                    help="Root directory where the model data will be placed")

    args = vars(ap.parse_args())

    deer_path = args['deer_images']
    background_path = args['background_images']
    tvt_split = args['train_val_test']
    tvt_split = tvt_split.split(",")  # 0-train, 1-val, 2-test
    model_datasets_dir = args['model_datasets_dir']

    # figure out how many files are in each of the deer/landscape datasets
    num_deer_files = _get_file_count(deer_path)
    num_landscape_files = _get_file_count(background_path)

    # use the minimum of the number of images and divide up into train/val/test split
    train_size = int(min(num_deer_files, num_landscape_files) * int(tvt_split[0]) / 100)
    val_size = int(min(num_deer_files, num_landscape_files) * int(tvt_split[1]) / 100)
    test_size = int(min(num_deer_files, num_landscape_files) * int(tvt_split[2]) / 100)

    print(f"Training Images: {train_size}, Validation Images: {val_size}, Test Images: {test_size}")

    # create datasets
    _create_dataset(train_size, val_size, test_size, deer_path, background_path, model_datasets_dir)


if __name__ == '__main__':
    _main()
