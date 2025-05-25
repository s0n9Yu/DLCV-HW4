# DLCV 2025 HW4
name: 楊宗儒 id: 111550098

## Introduction 

In this homework, we need to restore images degraded by rain and snow with a PromptIR based model.

## How to install

### Dataset

```
gdown 1bWW2kmK5RSuEaKwMRUWXoamYiQHQ2Yux
unzip data_ours.zip
```

### Environment Setup

```
conda env create -f environment.yml
conda activate promptir 
```

### Training

```
python3 train.py --batch_size 4 --epochs 500 --expname <expname>
```

The checkpoint will be stored in train_ckpt/

### Inference

```
python3 inference.py --ckpt_path <checkpoint path>
```

The prediction result would be in pred.npz

## Performance Snapshot

![](performance.png)
