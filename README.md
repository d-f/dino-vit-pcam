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


Vision Transformers and DINO background information
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
[Vision Transformers](https://arxiv.org/abs/2010.11929) are similar to the transformers used in natural language processing, but with minimal adaptions to allow for images to be used as input. Images divided into patches which are flattened into sequences of pixels and are treated as word tokens are in NLP. Just like NLP transformers, vision transformers are essentially entirely made from attention layers, and don't make use of any convolutional layers.
![image](https://github.com/d-f/dino-vit-pcam/assets/118086192/9c6ef466-166b-4f5d-8161-e2b1f0662ccc)


Figure 1: Vision Transformer architecture (image taken from [paper](https://arxiv.org/abs/2010.11929))

Self distillation with no labels (DINO) was brought about by [Caron et al.](https://arxiv.org/abs/2104.14294) and allows models to warm up to large unlabeled datasets before training on them. In some cases the model is so sensitive to the relevant features in a dataset that only a simple linear classifier or KNN classifier needs to be used on top of the extracted features to allow for useful classification. DINO works by using two networks (a teacher and a student) with identical architecture, makes two different transformations of an image and trains the two models to have a similar output. This forces the models to learn a useful representation of the input and doesn't require the input to have a classification label. The parameters of the teacher aren't optimized with gradient decent but are averaged over a batch and copied from the student network as an exponential moving average. DINO avoids collapse of the two models just outputting the same thing by centering and sharpening the output of the teacher. More details can be found in the [paper](https://arxiv.org/abs/2104.14294).

![image](https://github.com/d-f/dino-vit-pcam/assets/118086192/b6a20929-d168-4ff2-8db1-b5fa9e22256d)

Figure 2: DINO illustration (taken from [paper](https://arxiv.org/abs/2104.14294))

