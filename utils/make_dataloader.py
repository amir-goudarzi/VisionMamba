'''
Taken originally from https://github.com/MalumaDev and modified

'''


import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import webdataset as wds
import wget
from torch.utils.data import Dataset
from tqdm import tqdm


def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def sudoku_dataset(path, tr_va_te="train", transform=None, type=4, split=None, return_whole_puzzle=False):
    transform = transform
    type = type
    path_out = Path(path) / f"offline_{tr_va_te}_{split}.tar"
    samples_cells = []
    samples_pixels = []
    samples_labels = []
    if not path_out.exists():
        files = os.walk(os.path.join(path))
        for root, dirs, files in tqdm(files, desc="Generating dataset"):
            for f in files:
                if "numTrain::00050" in os.path.join(root, f) and "overlap::0.00" in os.path.join(root, f):
                    if split is None or ("split::0" + str(split + 1)) in os.path.join(root, f) or (
                            "split::" + str(split + 1)) in os.path.join(root, f):
                        if tr_va_te + "_puzzle_pixels" in f:
                            with open(os.path.join(root, f), "r") as liner:
                                for i, l in enumerate(liner.readlines()):
                                    pixels = []
                                    for c in range(type * type):
                                        step = 28 * 28
                                        number = l.split("\t")[c * step:c * step + step]
                                        pixels.append([float(n) for n in number])
                                    image = np.zeros((28 * type, 28 * type))
                                    for i in range(type):
                                        for j in range(type):
                                            image[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = np.reshape(
                                                np.array(pixels[i * type + j]), (28, 28))
                                    samples_pixels.append(image)
                                    if (tr_va_te == "valid" or tr_va_te == "test") and i >= 100:
                                        break

                        if tr_va_te + "_cell_labels" in f:
                            with open(os.path.join(root, f), "r") as liner:
                                for i, l in enumerate(liner.readlines()):
                                    cells = [int(j.split("_")[1]) - (0 if not "EMNIST" in str(path_out) else 11) for c, j in enumerate(l.split("\t"))]
                                    samples_cells.append(cells)
                                    if (tr_va_te == "valid" or tr_va_te == "test") and i >= 100:
                                        break

                        if tr_va_te + "_puzzle_labels" in f:
                            with open(os.path.join(root, f), "r") as liner:
                                for i, l in enumerate(liner.readlines()):
                                    label = 1 if l.split("\t")[0] == "1" else 0
                                    samples_labels.append(label)
                                    if (tr_va_te == "valid" or tr_va_te == "test") and i >= 100:
                                        break

        samples = [(p, c, l) for p, c, l in zip(samples_pixels, samples_cells, samples_labels)]
        with wds.TarWriter(str(path_out)) as dst:
            key = 0
            for pixels, cell, label in tqdm(samples):
                sample = {
                    "__key__": f"{key:08d}",
                    "png": pixels,
                    "cell.pyd": np.asarray(cell),
                    "cls": label
                }
                dst.write(sample)
                key += 1

    dataset = wds.WebDataset(str(path_out), shardshuffle=True, handler=wds.warn_and_continue).shuffle(
            100000 if tr_va_te == "train" else 0) \
            .decode("pil").to_tuple("jpg;png", "cell.pyd", "cls")

    if return_whole_puzzle:
        return dataset.map_tuple(
            lambda x: transform(x),  # Apply the transformation if provided
            None, None)
    else:
        return dataset.map_tuple(
            lambda x: image_to_sub_square(transform(x), type=type),
            None, None)



def image_to_sub_square(image, type=4):
    out = []
    for j in range(type):
        for i in range(type):
            out.append(image[:, j * 28: (j + 1) * 28, i * 28: (i + 1) * 28])
    return torch.cat(out, 0)


# def __len__(self):
#     return len(self.samples)
# def __getitem__(self, index):
#     item = self.samples[index]
#     imgs = []
#     for n in item[0]:
#         img = Image.fromarray(np.reshape(np.array(n), (28,28)), mode="L")
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         imgs.append(img)
#     imgs = torch.cat(imgs)
#     labels = torch.tensor(np.array(item[1]))
#     sudoku_label = item[2]
#     return imgs, labels, sudoku_label

dataset_link = {
    "mnist4": "https://linqs-data.soe.ucsc.edu/public/datasets/ViSudo-PC/v01/ViSudo-PC_dimension::4_datasets::mnist_strategy::simple.zip",
    "mnist9": "https://linqs-data.soe.ucsc.edu/public/datasets/ViSudo-PC/v01/ViSudo-PC_dimension::9_datasets::mnist_strategy::simple.zip",
    "emnist4": "https://linqs-data.soe.ucsc.edu/public/datasets/ViSudo-PC/v01/ViSudo-PC_dimension::4_datasets::emnist_strategy::simple.zip",
    "emnist9": "https://linqs-data.soe.ucsc.edu/public/datasets/ViSudo-PC/v01/ViSudo-PC_dimension::9_datasets::emnist_strategy::simple.zip",
    "fmnist4": "https://linqs-data.soe.ucsc.edu/public/datasets/ViSudo-PC/v01/ViSudo-PC_dimension::4_datasets::fmnist_strategy::simple.zip",
    "fmnist9": "https://linqs-data.soe.ucsc.edu/public/datasets/ViSudo-PC/v01/ViSudo-PC_dimension::9_datasets::fmnist_strategy::simple.zip",
    "kmnist4": "https://linqs-data.soe.ucsc.edu/public/datasets/ViSudo-PC/v01/ViSudo-PC_dimension::4_datasets::kmnist_strategy::simple.zip",
    "kmnist9": "https://linqs-data.soe.ucsc.edu/public/datasets/ViSudo-PC/v01/ViSudo-PC_dimension::9_datasets::kmnist_strategy::simple.zip",
}


def download_dataset(dataset, path):
    if not path.exists():
        print("Downloading data")
        path.mkdir(parents=True, exist_ok=True)
        wget.download(
            dataset_link[dataset],
            str(path / "data.zip"), bar=bar_progress)

        print("Extracting data")
        with zipfile.ZipFile(path / "data.zip", 'r') as zip_ref:
            zip_ref.extractall(path)
            # os.remove(path / "data.zip")


def get_loaders(batch_size, type="mnist4", split=10, num_workers=12, path='.', return_whole_puzzle=False):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    path = Path(path)
    match type:
        case 'mnist4':
            path = path / Path("./data/MNISTx4Sudoku")
            download_dataset(type, path)

            transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            n_classes = 4

        case 'emnist4':
            path = Path("./data/EMNISTx4Sudoku")
            download_dataset(type, path)

            n_classes = 4
        case "fmnist4":
            path = Path("./data/FMNISTx4Sudoku")
            download_dataset(type, path)

            n_classes = 4
        case "kmnist4":
            path = Path("./data/KMNISTx4Sudoku")
            download_dataset(type, path)

            n_classes = 4

        case 'mnist9':
            path = Path("./data/MNISTx9Sudoku")
            download_dataset(type, path)

            n_classes = 9

        case 'emnist9':
            path = Path("./data/EMNISTx9Sudoku")
            download_dataset(type, path)

            n_classes = 9

        case "fmnist9":
            path = Path("./data/FMNISTx9Sudoku")
            download_dataset(type, path)

            n_classes = 9

        case "kmnist9":
            path = Path("./data/KMNISTx9Sudoku")
            download_dataset(type, path)

            n_classes = 9

        case _:
            raise ValueError(f"Dataset {type} not supported.")

    train_set = sudoku_dataset(path=path, tr_va_te="train",
                               transform=transform, type=n_classes, split=split, return_whole_puzzle=return_whole_puzzle)

    val_set = sudoku_dataset(path=path, tr_va_te="valid",
                             transform=transform, type=n_classes, split=split, return_whole_puzzle=return_whole_puzzle)

    testset = sudoku_dataset(path=path, tr_va_te="test",
                             transform=transform, type=n_classes, split=split, return_whole_puzzle=return_whole_puzzle)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              num_workers=num_workers, drop_last=True)

    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            num_workers=num_workers, drop_last=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             num_workers=num_workers, drop_last=True)

    return trainloader, valloader, testloader, n_classes