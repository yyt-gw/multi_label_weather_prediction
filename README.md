# Multi-label binary classification for multiple weathers prediction

- This repository is for multi-label binary classification problems like predicting multiple weather conditions from the scene of a given image.

<img src="./assets/sample.jpg" width="600" height="600">

## Requirements and setup

- Python version 3.10.5

- Please install required packages and dependencies by

```console
pip install -r requirements.txt
```

## Download dataset and pretrained weights

- Download [data](https://drive.google.com/file/d/18-FgRSJMg5DJuyhahAJeOuisjw35CPey/view?usp=share_link) and extract under the root of the repository
- Download [weights](https://drive.google.com/file/d/19zzUNvY4HlzkxLmgGP7nsgZuKrok-dCn/view?usp=share_link)and extract under the root of the repository

## How to train

```console
python train.py --evaluate
```

## How to validate and visualize results

```console
python validate.py
```
