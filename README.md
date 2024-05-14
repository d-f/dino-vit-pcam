# dino-vit-pcam

This repository makes use of vit-pytorch: [https://github.com/lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)

In order to run:
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
- Set up environment 
  - Download Anaconda [https://www.anaconda.com/download](https://www.anaconda.com/download)
  - Create a conda environment (in this case named dino_env)
```
conda create -n dino_env python=3
```
  - Clone this repository to any directory, in this case C:\\ml_code\\dino-vit-pcam\\
```
cd C:\ml_code\
git clone https://github.com/d-f/dino-vit-pcam.git
```
  - Download dependencies
```
cd dino-vit-pcam
pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html
```
  - Set up directories
    - Depending on OS, run either create_dirs.ps1 (for Windows) or create_dirs.sh (for Linux) and choose a "project directory" for everything to be added to, in this case "C:\\ml_projects\\fcn_segmentation\\"
```
C:\ml_code\dino-vit-pcam\create_dirs.ps1 "C:\\ml_projects\\dino_vit\\"
```
or  
```
bash C:\ml_code\dino-vit-pcam\create_dirs.sh
"/C/ml_projects/dino_vit/"
```
  - Train DINO (using hyperparameters included in [paper](https://arxiv.org/abs/2104.14294))
      - See create_argparser() on line 57 in pytorch_ViT_DINO.py for more details on adjustable hyperparameters
```
python pytorch_ViT_DINO.py -project_directory C:\ml_projects\dino-vit\ -model_save_name "dino_vit_model_1.pth.tar"
```
  - Fine-tune classifier
      - See create_argparser() on line 133 in pytorch_ViT_finetune.py for more details on adjustable hyperparameters
      - param_str set to "just_classifier" only trains a fully connected layer at the end of the network, "all" sets requires_grad to True for all parameters
```
python pytorch_ViT_finetune.py -project_directory C:\ml_projects\dino-vit -model_save_name "dino_vit_model_1.pth.tar" -param_str "just_classifer"
```
  - Calculate model performance (may need to add R to PATH)
```
Rscript C:\ml_code\dino_vit_pcam\utils\calc_model_performance.R
C:\ml_projects\dino_vit\results\dino_vit_model_1_test_results.json
C:\ml_projects\dino_vit\results\dino_vit_model_1_performance.json
```

For demonstration / debugging purposes a model was trained with DINO with a batch size of 120 and a learning rate of 3e-4 for 17 epochs, resulting in a loss value of 5.34.

A MLP classifier was added to this model and fine tuned for 4 epochs with a batch size of 16 and a learning rate of 1e-3, resulting in the following performance on the PCAM test dataset:

| Accuracy | Sensitivity (Recall) | Specificity |
| -------- | -------------------- | ----------- |
| 73.21%   | 75.80%               | 70.63%      |

The loss values throughout training indicate this model would benefit from more training during both phases of training, but these two short training phases were just used as a sanity check.

