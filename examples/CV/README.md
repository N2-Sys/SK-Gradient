# CV-Example

This is a reproducible example of applying SK-Gradient to computer vision tasks.

## Environment Requirement

The required packages are as follows:

+ Python 3.9.6
+ PyTorch 1.11.0
+ CUDA ToolKit 11.0
+ Ray 1.12.0
+ Pandas 1.4.2
+ NumPy 1.21.5 
+ Torchvision 0.11.3

## Datasets

We use the CIFAR-10 dataset provided by torchvision.datasets.
The download link is [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

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
python3 cv-resnet101.py/cv-vgg19.py
```

#### Stop the Ray runtime

```bash
ray stop
```