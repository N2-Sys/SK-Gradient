# CTR-Prediction-Example

This is a reproducible example of applying SK-Gradient to CTR prediction tasks.

## Environment Requirement

The required packages are as follows:

+ Python 3.9.6
+ PyTorch 1.11.0
+ CUDA ToolKit 11.0
+ Ray 1.12.0
+ Pandas 1.4.2
+ NumPy 1.21.5 
+ DeepCTR-torch 0.2.9

## Datasets

We use the Criteo_x1 dataset provided by [BARS](https://github.com/openbenchmark/BARS). The dataset statistics are summarized as follows:

| Dataset Split  | Total | #Train | #Validation | #Test | 
| :--------: | :-----: |:-----: | :----------: | :----: | 
| Criteo_x1 |  45,840,617     | 33,003,326   |  8,250,124     | 4,587,167     | 

[BARS](https://github.com/openbenchmark/BARS) provides two Python scripts to access and preprocess the dataset. We modify them to meet the requirements of our experiments:

`data/download.py` : This script downloads the complete Criteo_x1 dataset. The format of the dataset is libsvm.

`data/convert.py` : This script converts a libsvm dataset to csv format. Due to the large size of the complete dataset, the conversion requires large memory and a long time. We recommend setting a reasonable crop length on the first run for quick testing.

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