# dino-vit-pcam

Install vit-pytorch: [https://github.com/lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)
```
pip install vit-pytorch
```

Self distillation with no labels was brought about by the group _ and allows models to warm up to data before training them. In some cases the model is so tuned to the dataset that only a simple linear classifier or KNN classifier needs to be used on top of the extracted features to allow for useful classification. DINO works by using two networks (a teacher and a student) with identical architecture, makes two random transformations of an input and trains the two models to have a similar output. This forces the models to learn a useful representation of the input and doesn't require the input to have a classification label. The parameters of the teacher aren't optimized with gradient decent but are averaged over a batch and copied from the student network as an exponential moving average. DINO avoids collapse of the two models just outputting the same thing by centering and sharpening the output of the teacher.
