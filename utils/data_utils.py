import torchvision
import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path


def create_dino_datasets(dataset_root):
    train_dataset = torchvision.datasets.PCAM(
        split="train", root=dataset_root, download=False, transform=torchvision.transforms.ToTensor()
    )
    return train_dataset


def create_ft_datasets(dataset_root):
    train_dataset = torchvision.datasets.PCAM(
        split="train", root=dataset_root, download=False, transform=torchvision.transforms.ToTensor()
    )
    val_dataset = torchvision.datasets.PCAM(
        split="val", root=dataset_root, download=False, transform=torchvision.transforms.ToTensor()
    )
    test_dataset = torchvision.datasets.PCAM(
        split="test", root=dataset_root, download=False, transform=torchvision.transforms.ToTensor()
    )
    return train_dataset, val_dataset, test_dataset


def create_dino_dataloaders(train_dataset, batch_size):
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader


def create_ft_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def save_dino_results(args, optimizer):
    hyperparameters = {'batch size': args.batch_size,
                       'model save name': args.model_save_name,
                       'optimizer': optimizer.defaults,
                       'patience': args.patience,
                       'patch size': args.patch_size,
                       'linear transformation output dimension': args.dim,
                       'number of transformer blocks': args.depth,
                       'number of heads in attention layer': args.heads,
                       'dimension of mlp layer': args.mlp_dim,
                       'hidden layer': args.hidden_layer,
                       'projection network hidden dimension': args.projection_hidden_size,
                       'projection network number of layers': args.projection_layers,
                       'output logits dimension (K)': args.num_classes_K,
                       'student temperature': args.student_temp,
                       'teacher temperature': args.teacher_temp,
                       'upper bound for local crop': args.local_upper_crop_scale,
                       'lower bound for global crop': args.global_lower_crop_scale,
                       'moving average decay': args.moving_average_decay,
                       'center moving average decay': args.center_moving_average_decay}

    with open(Path(args.project_directory).joinpath('results_and_models').joinpath(
            f'{args.model_save_name[:-8]}_hyperparameters.json'), mode='w') as outfile:
        json.dump(hyperparameters, outfile)

def save_ft_results(args, optimizer):
    hyperparameters = {'batch size': args.batch_size,
                       'model save name': args.model_save_name,
                       'optimizer': optimizer.defaults,
                       'patience': args.patience,
                       'patch size': args.patch_size,
                       'linear transformation output dimension': args.dim,
                       'number of transformer blocks': args.depth,
                       'number of heads in attention layer': args.heads,
                       'dimension of mlp layer': args.mlp_dim,
                       }

    with open(Path(args.project_directory).joinpath('results_and_models').joinpath(
            f'{args.model_save_name[:-8]}_hyperparameters.json'), mode='w') as outfile:
        json.dump(hyperparameters, outfile)
        