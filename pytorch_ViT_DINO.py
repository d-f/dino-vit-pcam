import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from torchsummary import summary
from tqdm import tqdm
import argparse
from vit_pytorch import ViT, Dino
from pathlib import Path
import numpy as np
import os
import csv
import json


class ImageClassificationDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image = io.imread(Path(self.root_dir).joinpath('bin').joinpath(self.annotations.iloc[index, 0]))
        if self.transform:
            image = self.transform(image)

        return image


def save_checkpoint(state, filepath):
    print('saving...')
    torch.save(state, filepath)


def create_dataset(args):
    # create datasets
    train_dataset = ImageClassificationDataset(
        csv_file=Path(args.project_directory).joinpath('datasets').joinpath(args.train_filename),
        root_dir=args.project_directory,
        transform=transforms.ToTensor()
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader


def train(args, train_loader, learner, optimizer, device):
    best_loss = np.inf
    patience_counter = 0
    # if loss log file exists, remove to not include previous training runs
    if os.path.exists(Path(args.project_directory).joinpath('results_and_models').joinpath(
            f'{args.model_save_name[:-8]}_dino_loss_values.csv')):
        os.remove(Path(args.project_directory).joinpath('results_and_models').joinpath(
            f'{args.model_save_name[:-8]}_dino_loss_values.csv'))
    for epoch in range(args.num_epochs):
        if patience_counter == args.patience:
            break
        losses = []
        checkpoint = {'state_dict': learner.state_dict(), 'optimizer': optimizer.state_dict()}
        for batch_idx, data in enumerate(tqdm(train_loader)):
            data    = data.to(device=device)
            loss  = learner(data)
            optimizer.zero_grad() # clear gradient information
            loss.backward() # calculate gradient
            optimizer.step()
            learner.update_moving_average()
            losses.append(loss.item())
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


def save_results(args, optimizer):
    hyperparameters = {'batch size': args.batch_size,
                       'model save name': args.model_save_name,
                       'optimizer': optimizer.defaults,
                       'train dataset': args.train_filename,
                       'patience': args.patience,
                       'patch size': args.patch_size,
                       'linear transformation output dimension': args.dim,
                       'number of transformer blocks': args.depth,
                       'number of heads in attention layer': args.heads,
                       'dimension of mlp layer': args.mlp,
                       'hidden layer': args.hl,
                       'projection network hidden dimension': args.proj_hidden,
                       'projection network number of layers': args.proj_layers,
                       'output logits dimension (K)': args.K_classes,
                       'student temperature': args.student_temp,
                       'teacher temperature': args.teacher_temp,
                       'upper bound for local crop': args.local_upper_crop_scale,
                       'lower bound for global crop': args.global_lower_crop_scale,
                       'moving average decay': args.moving_average_decay,
                       'center moving average decay': args.center_moving_average_decay}

    with open(Path(args.project_directory).joinpath('results_and_models').joinpath(
            f'{args.model_save_name[:-8]}_hyperparameters.json'), mode='w') as outfile:
        json.dump(hyperparameters, outfile)


def main():

    parser = argparse.ArgumentParser()

    # directory where all relevant folders are located
    parser.add_argument("-dir", "--project_directory", type=str)
    # number of epochs to train for
    parser.add_argument("-epochs", "--num_epochs", type=int, default=300)
    # number of classes to predict between
    parser.add_argument("-classes", "--num_classes", type=int, default=2)
    # proportion to weight parameter update by
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    # number of epochs trained past when the loss decreases to a minimum
    parser.add_argument("-patience", "--patience", type=int, default=5)
    # number of inputs before gradient is calculated
    parser.add_argument("-batch_size", "--batch_size", type=int, default=120)
    # filename of the model, including .pth.tar
    parser.add_argument("-save", "--model_save_name", type=str)
    # name of csv with tile info
    parser.add_argument("-train_filename", "--train_filename", type=str, default='benbep_final_train.csv')
    # channels first
    parser.add_argument("-img_shape", "--img_shape", default=(3, 512, 512), type=tuple, nargs="+")
    # size of image patch, 8, 16 and 32 are good values
    parser.add_argument("-patch", "--patch_size", type=int, default=8)
    # last dimension of output tensor after linear transformation
    parser.add_argument("-dim", "--dim", type=int, default=1024)  # 1024
    # number of transformer blocks
    parser.add_argument("-depth", "--depth", type=int, default=6)  # 6
    # number of heads in multi-head attention layer
    parser.add_argument("-heads", "--heads", type=int, default=8)  # 8
    # dimension of multilayer perceptron layer
    parser.add_argument("-mlp", "--mlp_dim", type=int, default=2048)  # 2048
    # hidden layer name or index, from which to extract the embedding
    parser.add_argument("-hl", "--hidden_layer", type=str, default='to_latent')
    # projector network hidden dimension
    parser.add_argument("-proj_hidden", "--projection_hidden_size", type=int, default=512)
    # number of layers in projection network
    parser.add_argument("-proj_layers", "--projection_layers", type=int, default=4)
    # output logits dimensions (referenced as K in paper)
    parser.add_argument("-K_classes", "--num_classes_K", type=int, default=65336)
    # student temperature
    parser.add_argument("-stu_temp", "--student_temp", type=float, default=0.9)
    # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
    parser.add_argument("-teach_temp", "--teacher_temp", type=float, default=0.04)
    # upper bound for local crop - 0.4 was recommended in the paper
    parser.add_argument("-local_scale", "--local_upper_crop_scale", type=float, default=0.4)
    # lower bound for global crop - 0.5 was recommended in the paper
    parser.add_argument("-global_scale", "--global_lower_crop_scale", type=float, default=0.5)
    # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
    parser.add_argument("-ma_decay", "--moving_average_decay", type=float, default=0.9)
    # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
    parser.add_argument("-cma_decay", "--center_moving_average_decay", type=float, default=0.9)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else print('where did you last put the GPU?'))

    model = ViT(
        image_size=args.img_shape[1],
        patch_size=args.patch_size, # 16 x 16
        num_classes=args.num_classes,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads, # 6
        mlp_dim=args.mlp_dim
    )

    # data augmentation the same as original paper

    learner = Dino(
        model,
        image_size=args.img_shape[1],
        hidden_layer=args.hidden_layer,
        projection_hidden_size=args.projection_hidden_size,
        projection_layers=args.projection_layers,
        num_classes_K=args.num_classes_K,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
        local_upper_crop_scale=args.local_upper_crop_scale,
        global_lower_crop_scale=args.global_lower_crop_scale,
        moving_average_decay=args.moving_average_decay,
        center_moving_average_decay=args.center_moving_average_decay,
    )

    optimizer = optim.Adam(learner.parameters(), lr=args.learning_rate)
    learner.to(device)
    print(summary(learner, args.img_shape))
    train_loader = create_dataset(args=args)
    train(args=args, train_loader=train_loader, learner=learner, optimizer=optimizer, device=device)
    save_results(args=args, optimizer=optimizer)


if __name__ == '__main__':
    main()
