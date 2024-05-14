import torchvision
import torch
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path
from typing import Tuple, List
import numpy as np


def create_dino_datasets(dataset_root: str) -> Dataset:
    train_dataset = torchvision.datasets.PCAM(
        split="train", root=dataset_root, download=True, transform=torchvision.transforms.ToTensor()
    )
    return train_dataset


def create_ft_datasets(dataset_root: str) -> Tuple:
    train_dataset = torchvision.datasets.PCAM(
        split="train", root=dataset_root, download=True, transform=torchvision.transforms.ToTensor()
    )
    val_dataset = torchvision.datasets.PCAM(
        split="val", root=dataset_root, download=True, transform=torchvision.transforms.ToTensor()
    )
    test_dataset = torchvision.datasets.PCAM(
        split="test", root=dataset_root, download=True, transform=torchvision.transforms.ToTensor()
    )
    return train_dataset, val_dataset, test_dataset


def create_dino_dataloaders(train_dataset: Dataset, batch_size: int) -> DataLoader:
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader


def create_ft_dataloaders(
    train_dataset: Dataset, 
    val_dataset: Dataset, 
    test_dataset: Dataset, 
    batch_size: int
    ) -> Tuple:
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def save_dino_results(
        optimizer: torch.optim.Adam,
        batch_size: int,
        model_save_name: str,
        patience: int,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        hidden_layer: str,
        projection_hidden_size: int,
        projection_layers: int,
        num_classes_K: int,
        student_temp: float,
        teacher_temp: float,
        local_upper_crop_scale: float,
        global_lower_crop_scale: float,
        moving_average_decay: float,
        center_moving_average_decay: float,
        project_directory: Path
        ) -> None:
    hyperparameters = {'batch size': batch_size,
                       'model save name': model_save_name,
                       'optimizer': optimizer.defaults,
                       'patience': patience,
                       'patch size': patch_size,
                       'linear transformation output dimension': dim,
                       'number of transformer blocks': depth,
                       'number of heads in attention layer': heads,
                       'dimension of mlp layer': mlp_dim,
                       'hidden layer': hidden_layer,
                       'projection network hidden dimension': projection_hidden_size,
                       'projection network number of layers': projection_layers,
                       'output logits dimension (K)': num_classes_K,
                       'student temperature': student_temp,
                       'teacher temperature': teacher_temp,
                       'upper bound for local crop': local_upper_crop_scale,
                       'lower bound for global crop': global_lower_crop_scale,
                       'moving average decay': moving_average_decay,
                       'center moving average decay': center_moving_average_decay}

    with open(Path(project_directory).joinpath('results').joinpath(
            f'{model_save_name[:-8]}_hyperparameters.json'), mode='w') as outfile:
        json.dump(hyperparameters, outfile)


def save_ft_results(
        optimizer: torch.optim.Adam,
        batch_size: int,
        model_save_name: str,
        patience: int,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        project_directory: Path
        ) -> None:
    hyperparameters = {'batch size': batch_size,
                       'model save name': model_save_name,
                       'optimizer': optimizer.defaults,
                       'patience': patience,
                       'patch size': patch_size,
                       'linear transformation output dimension': dim,
                       'number of transformer blocks': depth,
                       'number of heads in attention layer': heads,
                       'dimension of mlp layer': mlp_dim,
                       }

    with open(Path(project_directory).joinpath('results').joinpath(
            f'{model_save_name[:-8]}_hyperparameters.json'), mode='w') as outfile:
        json.dump(hyperparameters, outfile)


def save_test_results(
    label_list: List, 
    prediction_list: List, 
    label_prob_list: List, 
    pred_prob_list: List,
    result_fp: Path
    ) -> None:
    result_dict = {
        "labels": label_list,
        "predictions": prediction_list,
        "label probabilities": [(int(x[0]), int(x[1])) for x in label_prob_list],
        "predictive probabilities": [(np.float64(x[0]), np.float64(x[1])) for x in pred_prob_list]
    }
    with open(result_fp, mode="w") as opened_json:
        json.dump(result_dict, opened_json)
