import torch
from vit_pytorch import ViT, Dino
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple


def save_checkpoint(state: Dict, filepath: Path) -> None:
    print('saving...')
    torch.save(state, filepath)


def define_device() -> torch.device:
    return torch.device('cuda')


def get_num_params(tensor_size: Tuple) -> int:
    params = 1
    for size_idx in tensor_size:
        params *= size_idx
    return params


def define_model(
    img_shape: Tuple, 
    patch_size: int, 
    num_classes: int, 
    dim: int, 
    depth: int, 
    heads: int, 
    mlp_dim: int
    ) -> ViT:
    return ViT(
        image_size=img_shape[1],
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim
    )


def define_learner(
        model: ViT, 
        img_shape: Tuple, 
        hidden_layer: str, 
        projection_hidden_size: int, 
        projection_layers: int, 
        num_classes_K: int, 
        student_temp: float,
        teacher_temp: float,
        local_upper_crop_scale: float,
        global_lower_crop_scale: float,
        moving_average_decay: float,
        center_moving_average_decay: float
        ) -> Dino:
    return Dino(
        model,
        image_size=img_shape[1],
        hidden_layer=hidden_layer,
        projection_hidden_size=projection_hidden_size,
        projection_layers=projection_layers,
        num_classes_K=num_classes_K,
        student_temp=student_temp,
        teacher_temp=teacher_temp,
        local_upper_crop_scale=local_upper_crop_scale,
        global_lower_crop_scale=global_lower_crop_scale,
        moving_average_decay=moving_average_decay,
        center_moving_average_decay=center_moving_average_decay,
    )


def define_optimizer(learner: ViT, learning_rate: float) -> torch.optim.Adam:
    return optim.Adam(learner.parameters(), lr=learning_rate)


def print_model_summary(model: ViT | Dino) -> None:
    '''
    prints the parameters and parameter size
    should contain an equal number of trainable and non-trainable
    parameters since the teacher network parameters are not updated
    with gradient descent
    '''
    trainable = 0
    non_trainable = 0
    for param in model.named_parameters():
        if param[1].requires_grad:
            trainable += get_num_params(param[1].size())
        else:
            non_trainable += get_num_params(param[1].size())
        print(param[0], param[1].size(), param[1].requires_grad)

    print("trainable parameters:", trainable)
    print("non-trainable parameters:", non_trainable)


def load_model(weight_path: Path, model: ViT) -> ViT:
    '''
    loads all parameters of a model
    :param weight_path: path to the .pth.tar file with parameters to update
    :param model: model object
    :return: the model with updated parameters
    '''
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def freeze_params(model: ViT, param_str: str) -> ViT:
    if param_str == "just_classifier":
        for param in model.named_parameters():
            if "mlp" in param[0]:
                param[1].requires_grad = True
            else:
                param[1].requires_grad = False
    elif param_str == "all":
        for param in model.parameters():
            param.requires_grad = True
    return model


def define_criterion() -> torch.nn.CrossEntropyLoss:
    return torch.nn.CrossEntropyLoss()
