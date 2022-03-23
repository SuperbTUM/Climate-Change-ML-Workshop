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
python trainer.py --batch_size 50 --tta
```



## Leaderboard

Last update on Mar. 22nd

| Accuracy | Accuracy Rank | F1 score | F1 Rank | Precision | Recall |
| -------- | ------------- | -------- | ------- | --------- | ------ |
| 0.8224   | 1             | 0.7750   | 2       | 0.8286    | 0.7871 |



## Issues

Attempts of tricks from [this paper](https://arxiv.org/pdf/1812.01187.pdf) are all failed. (?)

ResNet-101 is not applicable because of over-fitting issues.
