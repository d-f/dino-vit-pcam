import torch
from skimage import io
from tqdm import tqdm
import argparse
import os
import csv
from vit_pytorch import Dino
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from utils.data_utils import *
from utils.model_utils import *


def train(
        train_loader: DataLoader, 
        learner: Dino, 
        optimizer: optim, 
        device: torch.device, 
        project_directory: Path, 
        model_save_name: str, 
        num_epochs: int
        ) -> None:
    # if loss log file exists, remove to not include previous training runs
    res_file_path = Path(project_directory).joinpath('results_and_models').joinpath(f'{model_save_name[:-8]}_dino_loss_values.csv')
    
    if os.path.exists(res_file_path):
        os.remove(res_file_path)
    for epoch in range(num_epochs):
        losses = []
        for data in tqdm(train_loader):
            data    = data[0].to(device=device) # [0]: tensors [1]: labels
            loss  = learner(data)
            optimizer.zero_grad() # clear gradient information
            loss.backward() # calculate gradient
            optimizer.step()
            learner.update_moving_average()
            losses.append(loss.item())
        
        with torch.no_grad():        
            total_loss = sum(losses) / len(losses)

            print('epoch', epoch, 'loss: ', total_loss)

            with open(res_file_path, mode="a", newline="") as data:
                csv_writer = csv.writer(data)
                csv_writer.writerow((epoch, total_loss))
    checkpoint = {'state_dict': learner.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(state=checkpoint, filepath=project_directory.joinpath("models").joinpath(model_save_name))


def create_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # directory where all relevant folders are located
    parser.add_argument("-project_directory", type=Path)
    # directory where the PCAM data are located 
    parser.add_argument("-data_root", type=Path)
    # number of epochs to train for
    parser.add_argument("-num_epochs", type=int, default=18)
    # number of classes to predict between
    parser.add_argument("-num_classes", type=int, default=2)
    # proportion to weight parameter update by
    parser.add_argument("-learning_rate", type=float, default=3e-4)
    # number of epochs trained past when the loss decreases to a minimum
    parser.add_argument("-patience", type=int, default=5)
    # number of inputs before gradient is calculated
    parser.add_argument("-batch_size", type=int, default=120) 
    # filename of the model, including .pth.tar
    parser.add_argument("-model_save_name", type=str)
    # channels first
    parser.add_argument("-img_shape", default=(3, 96, 96), type=tuple, nargs="+")
    # size of image patch, 8, 16 and 32 are good values
    parser.add_argument("-patch_size", type=int, default=16)
    # last dimension of output tensor after linear transformation
    parser.add_argument("-dim", type=int, default=1024)
    # number of transformer blocks
    parser.add_argument("-depth", type=int, default=6)
    # number of heads in multi-head attention layer
    parser.add_argument("-heads", type=int, default=8)
    # dimension of multilayer perceptron layer
    parser.add_argument("-mlp_dim", type=int, default=2048)
    # hidden layer name or index, from which to extract the embedding
    parser.add_argument("-hidden_layer", type=str, default='to_latent')
    # projector network hidden dimension
    parser.add_argument("-projection_hidden_size", type=int, default=512)
    # number of layers in projection network
    parser.add_argument("-projection_layers", type=int, default=4)
    # output logits dimensions (referenced as K in paper)
    parser.add_argument("-num_classes_K", type=int, default=65336)
    # student temperature
    parser.add_argument("-student_temp", type=float, default=0.9)
    # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
    parser.add_argument("-teacher_temp", type=float, default=0.04)
    # upper bound for local crop - 0.4 was recommended in the paper
    parser.add_argument("-local_upper_crop_scale", type=float, default=0.4)
    # lower bound for global crop - 0.5 was recommended in the paper
    parser.add_argument("-global_lower_crop_scale", type=float, default=0.5)
    # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
    parser.add_argument("-moving_average_decay", type=float, default=0.9)
    # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
    parser.add_argument("-center_moving_average_decay", type=float, default=0.9)
    return parser.parse_args()


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
    learner = define_learner(
        model=model,
        img_shape=args. img_shape,
        hidden_layer=args.hidden_layer,
        projection_hidden_size=args.projection_hidden_size,
        projection_layers=args.projection_layers,
        num_classes_K=args.num_classes_K,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
        local_upper_crop_scale=args.local_upper_crop_scale,
        global_lower_crop_scale=args.global_lower_crop_scale,
        moving_average_decay=args.moving_average_decay,
        center_moving_average_decay=args.center_moving_average_decay
    )
    optimizer = define_optimizer(learner=learner, learning_rate=args.learning_rate)
    learner.to(device)
    print_model_summary(model=learner)
    train_dataset = create_dino_datasets(dataset_root=args.data_root)
    train_dataloader = create_dino_dataloaders(
        train_dataset=train_dataset,
        batch_size=args.batch_size
    )
    train(
        train_loader=train_dataloader, 
        learner=learner, 
        optimizer=optimizer, 
        device=device, 
        project_directory=args.project_directory,
        model_save_name=args.model_save_name,
        num_epochs=args.num_epochs
        )
    save_dino_results(
        optimizer=optimizer,
        batch_size=args.batch_size,
        model_save_name=args.model_save_name, 
        patience=args.patience,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
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
        project_directory=args.project_directory
        )


if __name__ == '__main__':
    main()
