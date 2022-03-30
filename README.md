## Introduction

This is the playground of [climate change workshop with machine learning](https://www.modelshare.org/detail/model:1535#) hosted by Columbia University. The challenge of this workshop is to deal with satellite image classification. The dataset is sampled from [BigEarthNet dataset](https://bigearth.net/). Full rules can be accessed [here](https://drive.google.com/file/d/1xAU_2IvoTVgmfwSGQ4lnvL68Tp520oHx/view).



## Quick Start

Create a new folder named `checkpoint` to store the trained model with the following command

```
mkdir checkpoint
```

Please pre-install the following packages as prerequisites

```
pip install aimodelshare-nightly dill pydot regex prefetch_generator tensorflow_datasets
```

Activate the program with the following command (But you can always change any hyper-parameters or call different backbones)

```
python trainer.py --batch_size 64 --tta --weight_decay 0.
```



## Leaderboard

Last update on Mar. 29th (EST)

| Accuracy | Accuracy Rank | F1 score | F1 Rank | Precision | Recall | Backbone |
|----------|---------------|----------|---------|-----------|--------|----------|
| 0.8444   | 1             | 0.8076   | 1       | 0.8443    | 0.8233 | ResNet50 |
| 0.8312   | 4             | 0.7861   | 6       | 0.8313    | 0.7973 | Xception |



## Tricks

This should be a hard classification problem, so there is no label smooth.

Dropout should be applied appropriately.

