import torch
from torch import nn
import utils.options as options
from utils.make_dataloader import get_loaders
from utils.scheduler import build_scheduler
import neptune.new as neptune
import os

# import argparse
# import datetime
# import numpy as np
# import time
# import torch.backends.cudnn as cudnn
# import json
import models_mamba


# from pathlib import Path

import timm 
# from timm.data import Mixup
from timm.models import create_model
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from timm.scheduler import create_scheduler
# from timm.optim import create_optimizer
# from timm.utils import NativeScaler, get_state_dict, ModelEma

# from datasets import build_dataset, build_cifar10_dataset
# from engine import train_one_epoch, evaluate
# from losses import DistillationLoss
# from samplers import RASampler
# from augment import new_data_aug_generator

# from contextlib import suppress

# import models_mamba


# log about
# import mlflow
print(timm.list_models(pretrained=True))

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n--- GPU Information ---\n")

    if torch.cuda.is_available():
        print(f"Model is using device: {device}")
        print(f"CUDA Device: {torch.cuda.get_device_name(device)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 2} MB")
    else:
        print("Model is using CPU")


    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=2,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=112
    ).to(device)

    params = {
        "MODEL": args.model,
        "NUM_PATCHES": int(112/model.patch_size),
        "NUM_LAYERS": model.depth,
        "EMBEDDING_DIM": model.embed_dim,
        "BATCH_SIZE": args.batch_size,
        "OPTIMIZER": args.optimizer,
        "LEARNING_RATE": args.learning_rate,
        "SCHEDULER_USED": args.scheduler,
        "NUM_EPOCHS": args.num_epochs,
        "NUM_WORKERS": args.num_workers
    }

    train_loader, val_loader, test_loader, n_classes = get_loaders(batch_size= params['BATCH_SIZE'], 
                                                                   num_workers=params['NUM_WORKERS'], 
                                                                   path= os.path.join(args.dir, args.dataset_path), 
                                                                   split=args.split,
                                                                   return_whole_puzzle=True)
    print("\n ---Dataloaders succusfully created--- \n")

    loss_fn = nn.CrossEntropyLoss()
    if args.optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr= args.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr= args.learning_rate)

    if args.scheduler:
        scheduler = build_scheduler(optimizer, lr=params["LEARNING_RATE"])
    
    run = neptune.init_run(
            project=args.neptune_project,
            api_token=args.neptune_api_token
        )
    run["parameters"] = params
    run["sys/group_tags"].add(["Mamba"])

    def train(epoch):

        correct = 0
        total = 0
        correct_train = 0
        total_train = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x, _, sudoku_label = batch
            x, sudoku_label = x.to(device), sudoku_label.to(device)
            y = model(x)
            loss = loss_fn(y, sudoku_label)
            loss.backward()
            optimizer.step()
            predictions = torch.argmax(y, 1)
            total_train += x.shape[0]
            correct_train += predictions.eq(sudoku_label).sum().item()

            # Log training loss to Neptune
            run[f"train/loss"].log(loss.item())

            # print(f'Epoch {epoch+1}, Loss: {loss.item():.2f}')
        train_accuracy = ((correct_train / total_train) * 100)
        # print(f'\n\n-----Epoch {epoch+1}, Train accuracy: {train_accuracy:.2f}-----')
        run[f"train/accuracy"].log(train_accuracy)


        for batch_idx, batch in enumerate(val_loader):
            x, _, sudoku_label = batch
            x, sudoku_label = x.to(device), sudoku_label.to(device)
            y = model(x)
            loss = loss_fn(y, sudoku_label)
            predictions = torch.argmax(y, 1)
            total += x.shape[0]
            correct += predictions.eq(sudoku_label).sum().item()

            # Log validation loss to Neptune
            run[f"val/loss"].log(loss.item())

        acc = (correct / total) * 100

        # Log validation accuracy to Neptune
        run[f"val/accuracy"].log(acc)

        # print(f'-----Epoch {epoch+1}, Validation accuracy: {acc:.2f}-----\n\n')
        if args.scheduler:
            scheduler.step()

    def test():
        model.eval()
        correct = 0
        total = 0
        for batch in test_loader:
            x, _, sudoku_label = batch
            x, sudoku_label = x.to(device), sudoku_label.to(device)
            y = model(x)
            predictions = torch.argmax(y, 1)
            total += x.shape[0]
            correct += predictions.eq(sudoku_label).sum().item()

        acc = (correct / total) * 100

        # Log testing accuracy to Neptune
        run[f"test/accuracy"].log(acc)

        print(f'\n\nTest accuracy: {acc:.2f}\n\n')

    print("\n\n--Started Training--\n\n")

    for epoch in range(args.num_epochs):
        train(epoch)
    test()
    run.stop()


if __name__ == "__main__":
    args = options.read_command_line()
    main(args)