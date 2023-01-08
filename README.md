# SK-Gradient
Efficient Communication for Distributed Machine Learning with Data Sketch

## Introduction
SK-Gradient is a gradient compression scheme that solely builds on sketch. At its core, we propose a new sketch named FGC Sketch to implement fast gradient compression. With FGC Sketch, SK-Gradient is able to provide high compression ratios and low compression losses with low computational overhead.

This repo is our PyTorch implementation for SK-Gradient.

## Environment Requirement

The required packages are as follows:

+ Python 3.9.6
+ PyTorch 1.11.0
+ CUDA ToolKit 11.0

## Examples

We provide reproducible training scripts for all models in our experiments as examples. 
We also provide the download links or download scripts for the datasets in these examples.
Please refer to our [examples](https://github.com/N2-Sys/SK-Gradient/tree/main/examples) for details.