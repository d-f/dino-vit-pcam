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
  - Train DINO (using hyperparameters including in  paper) [https://arxiv.org/abs/2104.14294](https://arxiv.org/abs/2104.14294)
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


Vision Transformers and DINO background information
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
Self distillation with no labels was brought about by the group _ and allows models to warm up to data before training them. In some cases the model is so tuned to the dataset that only a simple linear classifier or KNN classifier needs to be used on top of the extracted features to allow for useful classification. DINO works by using two networks (a teacher and a student) with identical architecture, makes two random transformations of an input and trains the two models to have a similar output. This forces the models to learn a useful representation of the input and doesn't require the input to have a classification label. The parameters of the teacher aren't optimized with gradient decent but are averaged over a batch and copied from the student network as an exponential moving average. DINO avoids collapse of the two models just outputting the same thing by centering and sharpening the output of the teacher.
