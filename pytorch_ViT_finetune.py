import torchvision
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
# from torchsummary import summary
from tqdm import tqdm
import argparse
from vit_pytorch import ViT, Dino
from pathlib import Path
import numpy as np
import os
import csv
import json
from sklearn.preprocessing import OneHotEncoder
from utils.data_utils import *
from utils.model_utils import *


def train(args, train_loader, model, optimizer, device, val_loader, criterion):
    best_loss = np.inf
    patience_counter = 0
    # if loss log file exists, remove to not include previous training runs
    if os.path.exists(Path(args.project_directory).joinpath('results_and_models').joinpath(
            f'{args.model_save_name[:-8]}_finetune_loss_values.csv')):
        os.remove(Path(args.project_directory).joinpath('results_and_models').joinpath(
            f'{args.model_save_name[:-8]}_finetune_loss_values.csv'))
    for epoch in range(args.num_epochs):
        if patience_counter == args.patience:
            break
        losses = []
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        model.train()
        for data, labels in tqdm(train_loader):
            data    = data.to(device=device) 
            labels = labels.to(device=device)
            scores  = model(data)
            optimizer.zero_grad() # clear gradient information
            loss = criterion(scores, labels)
            loss.backward() # calculate gradient
            optimizer.step()
            with torch.no_grad():
                losses.append(loss.item())
        with torch.no_grad():        
            total_loss = sum(losses) / len(losses)
            if total_loss < best_loss:
                best_loss = total_loss
                save_checkpoint(
                state=checkpoint,
                filepath=Path(args.project_directory).joinpath('results_and_models').joinpath(args.model_save_name)
                )
                patience_counter = 0
            else:
                patience_counter += 1

            print('epoch', epoch, 'loss: ', total_loss, 'patience counter', patience_counter)

            if os.path.exists(Path(args.project_directory).joinpath('results_and_models').joinpath(f'{args.model_save_name[:-8]}_dino_loss_values.csv')) == False:
                with open(Path(args.project_directory).joinpath('results_and_models').joinpath(f'{args.model_save_name[:-8]}_dino_loss_values.csv'), mode="w", newline="") as data:
                    csv_writer = csv.writer(data)
                    csv_writer.writerow((epoch, total_loss))
            else:
                with open(Path(args.project_directory).joinpath('results_and_models').joinpath(f'{args.model_save_name[:-8]}_dino_loss_values.csv'), mode="a", newline="") as data:
                    csv_writer = csv.writer(data)
                    csv_writer.writerow((epoch, total_loss))


def test_best_model(
    weight_path: str, 
    num_classes: int, 
    loader: torch.utils.data.DataLoader, 
    model, 
    model_save_dir: Path, 
    device: torch.device
    ) -> tuple([list, list, list, list]):
    """
    evaluates best performing model on test set
    as dictated by best_checkpoint based on individual
    image tiles and performance based on correctly predicted
    slides when all predictions within a slide are averaged
    """
    with torch.no_grad():
        load_model(weight_path=weight_path, model=model)
        label_list = []
        prediction_list = []
        pred_prob_list = []
        label_prob_list = []
        model.eval()
        onehot_encoder = OneHotEncoder(sparse=False)
        # fit encoder to a list of all classes from 0 to num_classes - 1
        # reshaped from [0, 1, 2, 3] to [[0], [1], [2], [3]]
        onehot_encoder = onehot_encoder.fit(
            np.array([x for x in range(num_classes)]).reshape(-1, 1)
        )

        for imgs, labels in tqdm(loader, desc="Test"):
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            scores = model(imgs)
            label_list += [int(x) for x in labels]
            prediction_list += [int(np.argmax(x)) for x in scores]
            pred_prob_list += [tuple(x.cpu().numpy().astype(np.float64)) for x in scores]
            # labels is a list of tensors, so they need to be placed onto the cpu, converted to numpy arrays, 
            # reshaped with .reshape(-1, 1) to go from [1, 0] to [[1, 0]], transformed with onehot encoder, 
            # and indexed with [0] to go from [[0, 1]] to [0, 1]. then converted to a tuple to be JSON serialized
            label_prob_list += [tuple(onehot_encoder.transform(x.cpu().numpy().reshape(-1, 1))[0]) for x in labels]

        return label_list, prediction_list, label_prob_list, pred_prob_list


def create_argparser():
    parser = argparse.ArgumentParser()
    # directory where all relevant folders are located
    parser.add_argument("-project_directory", type=str, default="C:\\personal_ML\\DINOVIT_PCAM\\")
    # number of epochs to train for
    parser.add_argument("-num_epochs", type=int, default=300)
    # number of classes to predict between
    parser.add_argument("-num_classes", type=int, default=2)
    # proportion to weight parameter update by
    parser.add_argument("-learning_rate", type=float, default=1e-4)
    # number of epochs trained past when the loss decreases to a minimum
    parser.add_argument("-patience", type=int, default=5)
    # number of inputs before gradient is calculated
    parser.add_argument("-batch_size", type=int, default=16)
    # filename of the model, including .pth.tar
    parser.add_argument("-model_save_name", type=str, default="test_model.pth.tar")
    # channels first
    parser.add_argument("-img_shape", default=(3, 96, 96), type=tuple, nargs="+")
    # size of image patch, 8, 16 and 32 are good values
    parser.add_argument("-patch_size", type=int, default=16) # 8
    # last dimension of output tensor after linear transformation
    parser.add_argument("-dim", type=int, default=1024)  # 1024
    # number of transformer blocks
    parser.add_argument("-depth", type=int, default=6)  # 6
    # number of heads in multi-head attention layer
    parser.add_argument("-heads", type=int, default=8)  # 8
    # dimension of multilayer perceptron layer
    parser.add_argument("-mlp_dim", type=int, default=2048)  # 2048
    parser.add_argument("-weight_path", type=Path)
    parser.add_argument("-param_str", choices=["just_classifier", "all"], default="just_classifier") # decides which layers to train
    return parser.parse_args()


def validate_model(
    model, 
    val_loader, 
    device, 
    criterion, 
    val_losses, 
    num_correct_val
    ):
    """
    evaluates model performance on the validation set
    """
    with torch.no_grad():
        model.eval()  # turn dropout and batch norm off
        for val_data, val_targets in tqdm(val_loader, desc='Validation'):
            val_data = val_data.to(device=device)
            val_targets = val_targets.to(device=device)
            val_scores = model(val_data)
            _, val_predictions = val_scores.max(1)
            num_correct_val += (val_predictions == val_targets).sum().detach()
            val_loss = criterion(val_scores, val_targets)
            val_losses.append(val_loss.item())
        return val_losses, num_correct_val


def main():
    args = create_argparser()
    device = define_device()
    model = define_model(
        img_shape=args.img_shape,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        dim=args.dim, 
        depth=args.depth, 
        mlp_dim=args.mlp_dim,
        heads=args.heads
    )
    criterion = define_criterion()
    optimizer = define_optimizer(learner=model, learning_rate=args.learning_rate)
    model.to(device)
    model = freeze_params(model=model, param_str=args.param_str)
    print_model_summary(model=model)
    train_dataset, val_dataset, test_dataset = create_ft_datasets(dataset_root="C:\\Users\\danan\\protean\\PCAM\\")
    train_dataloader, val_dataloader, test_dataloader = create_ft_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size
    )
    train(args=args, train_loader=train_dataloader, model=model, optimizer=optimizer, device=device, val_loader=val_dataloader, criterion=criterion)
    save_ft_results(args=args, optimizer=optimizer)
    test_best_model(
        weight_path=args.weight_path,
        num_classes=args.num_classes, 
        loader=test_dataloader, 
        model=model, 
        model_save_dir=args.project_directory.joinpath("models"),
        device=device
    )


if __name__ == '__main__':
    main()