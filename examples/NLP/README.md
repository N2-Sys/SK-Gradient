# NLP-Example

This is a reproducible example of applying SK-Gradient to NLP tasks.

## Environment Requirement

The required packages are as follows:

+ Python 3.9.6
+ PyTorch 1.11.0
+ CUDA ToolKit 11.0
+ Ray 1.12.0
+ Pandas 1.4.2
+ NumPy 1.21.5 
+ pytorch_pretrained_bert 0.6.2
+ scikit-learn 0.24.2

## Datasets

We use the [THUCNews](http://thuctc.thunlp.org/) dataset. More precisely, we use a small subset of the THUCNews dataset provided by this repository [Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch).

## Pretrained model

Before running our training script, we need to put the pre-trained model in the `bert_pretrain` directory. Please refer to the README in the `bert_pretrain` directory to download it.

## Usage

#### Start the Ray cluster

1. Start a head node
```bash
ray start --head --node-ip-address {your_head_node_ip_address}
```

2. Join this Ray runtime from other nodes

```bash
ray start --address {your_head_node_ip_address}:{node_port}
```

#### Submit the training task

```bash
python3 ctr-deepfm.py
```

#### Stop the Ray runtime

```bash
ray stop
```