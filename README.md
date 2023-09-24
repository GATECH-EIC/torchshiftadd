<p align="center">
    <img src="figs/logo_torchshiftadd.png" alt="torchshiftadd logo" width="500">
</p>

<h2 align="center">
    A PyTorch library for developing energy efficient multiplication-less models.
</h2>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](https://opensource.org/licenses/Apache-2.0)
[![Contributions](https://img.shields.io/badge/contributions-welcome-blue)](https://github.com/GATECH-EIC/torchshiftadd/blob/master/CONTRIBUTING.md)

# TorchShiftAdd Overview

Welcome to TorchShiftAdd, your go-to open-source library for crafting energy-efficient multiplication-less models and applications!

[TorchShiftAdd](https://github.com/GATECH-EIC/torchshiftadd) embodies a pioneering initiative to simplify and expand the realm of multiplication-less networks within the machine learning community. Key features include:

* Ready-to-use implementation of a wide range of ShiftAdd-based multiplication-less CNNs or Transformers.
* CUDA kernels and TVM compilation support for seamless GPU deployment.
* Profiling tools to furnish FLOPs, energy, and latency breakdown data for in-depth analysis and optimization.
* Hardware accelerator simulators to estimate energy savings and latency improvements on ASICs or FPGAs.
* Flexible support for developing both algorithmic and hardware accelerator designs tailored for multiplication-less networks.

<!-- <details><summary>List of Implemented Papers</summary><p> -->

## List of Implemented Papers
* **ShiftAdd-based Convolutional Neural Networks**
    + [[NeurIPS'20] ShiftAddNet: A Hardware-Inspired Deep Network](https://arxiv.org/abs/2010.12785)
    + [[CVPR'20 Oral] AdderNet: Do We Really Need Multiplications in Deep Learning?](https://arxiv.org/abs/1912.13200)
    + [[CVPR'21 Workshop] DeepShift: Towards Multiplication-Less Neural Networks](https://arxiv.org/abs/1905.13298)
    + [[ICLR'23] Bit-Pruning: A Sparse Multiplication-Less Dot-Product](https://openreview.net/pdf?id=YUDiZcZTI8)
* **ShiftAdd-based Transformers**
    + [[NeurIPS'23] ShiftAddViT: Mixture of Multiplication Primitives Towards Efficient Vision Transformer](https://arxiv.org/abs/2306.06446)
    + [[NeurIPS'22] EcoFormer: Energy-Saving Attention with Linear Complexity](https://arxiv.org/abs/2209.09004)
* **Linear Attention in Transformers**
    + [[CVPR'23] Castling-ViT: Compressing Self-Attention via Switching Towards Linear-Angular Attention During Vision Transformer Inference](https://arxiv.org/abs/2211.10526)
    + [[HPCA'23] ViTALiTy: Unifying Low-rank and Sparse Approximation for Vision Transformer Acceleration with a Linear Taylor Attention](https://arxiv.org/abs/2211.05109)
* **Hardware Accelerators for ShiftAdd-based Multiplication-less Networks**
    + [[ICCAD'22] NASA: Neural Architecture Search and Acceleration for Hardware Inspired Hybrid Networks](https://arxiv.org/abs/2210.13361)
    + [[IEEE TCAS-I] NASA+: Neural Architecture Search and Acceleration for Multiplication-Reduced Hybrid Networks](https://ieeexplore.ieee.org/document/10078392)

# Installation

Coming soon.

# Qucik Start

Coming soon.

# Contributing

TorchShiftAdd is released under [Apache-2.0 License](LICENSE). Everyone is welcome to contribute to the development of TorchShiftAdd. Please refer to [contributing guidelines](CONTRIBUTING.md) for more details.

# Acknowledgement

Coming soon.