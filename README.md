# Fast Point Cloud Diffusion (FPCD) - Summer Internship Project

## Overview

This repository showcases my work during my Summer Internship at the National Energy Research Scientific Computing Center (NERSC) at Lawrence Berkeley National Laboratory, under the mentorship of Vinicius Mikuni. The primary work revolves around the Fast Point Cloud Diffusion (FPCD) model, predominantly developed by Vinicius Mikuni. My contribution involves the development of an ODESolver and an SDESolver (Stochastic Differential Equation Solver), pivotal for generating samples using the FPCD model.

## Project and Research Focus

The core of our research is optimizing sampling techniques for particle physics datasets, with a focus on numerical methods and data analysis. These datasets present challenges such as continuous coordinates, stochastic dimensionality, and permutation invariance symmetries. Traditional deep generative models, typically used for images, may not suit these datasets. Our approach, employing Fast Point Cloud Diffusion (FPCD), aims to expedite diffusion models and enhance sampling efficiency. The efficacy of our model is measured by reduced sampling time and minimized Wasserstein Distances between the generated and true data distributions.

## Resources

- [Stochastic Solver Implementation](https://github.com/hasifnumerics/GSGM-SDE-Solver/blob/33d4f1fffc32bc26cf91d4e80cb87171654f0fe0/scripts/continuous/GSGM_uniform_SDESolver.py)
- [Project Report](https://github.com/hasifnumerics/GSGM-SDE-Solver/blob/8e232eca9d1bcac0d7636d929adb1f1247e5237c/FPCD_Stochastic_Solver_Optimization_Report.pdf)
- [Research Poster](https://github.com/hasifnumerics/GSGM-SDE-Solver/blob/a4c9a22c17c69a47b210b0e38ca23d94639b3e60/Hasif%20Poster%20updated.pdf)

## References

1. Mikuni et al. 2023, Fast Point Cloud Generation with Diffusion Models in High Energy Physics.
2. Song et al. 2020, Score-Based Generative Modeling through Stochastic Differential Equations.

## Fast Point Cloud Diffusion Implementation

![Visualization of FPCD](./assets/plot_2D.png)

This is the official implementation of the FPCD paper, leveraging a diffusion model to generate particle jets. Progressive distillation is employed to expedite generation, as detailed in [this paper](https://arxiv.org/abs/2202.00512).

## Docker Container

Access our docker image [here](https://hub.docker.com/layers/vmikuni/tensorflow/ngc-22.08-tf2-v0/images/sha256-2bfbd4e3af2564a1bd2d0660899a4d295d78eb015f1b1492119774817013670b?context=repo) and use the following commands for setup:

```bash
shifterimg -v pull vmikuni/tensorflow:ngc-22.08-tf2-v0
shifter --image=vmikuni/tensorflow:ngc-22.08-tf2-v0 --module=gpu,nccl-2.15



# Training a new model

To train a new model from scratch, first download the data with either [30 particles](https://zenodo.org/record/6975118) or [150 particles](https://zenodo.org/record/6975117).
The baseline model can be trained with:
```bash
cd scripts
python train.py [--big]
```
with optional --big flag to choose between the 30 or 150 particles dataset.
After training the baseline model, you can train the distilled models with:
```bash
python train.py --distill --factor 2
```
This step will train a model that decreases the overall number of time steps by a factor 2. Similarly, you can load the distilled model as the next teacher and run the training using ```--factor 4``` and so on to halve the number of evaluation steps during generation.

To reproduce the plots provided in the paper, you can run:
```bash
python plot_jet.py [--distill --factor 2] --sample
```
The command will generate new observations with optional flags to load the distilled models. Similarly, if you already have the samples generated and stored, you can omit the ```--sample``` flag to skip the generation.

# Plotting and Metrics

The calculation os the physics inspired metrics is taken directly from the [JetNet](https://github.com/jet-net/JetNet) repository, thus also need to be cloned. Notice that while our implementation is carried out using TensorFlow while the physics inspired metrics are implemented in Pytorch.

Out distillation model is partially based on a [Pytorch implementation](https://github.com/Hramchenko/diffusion_distiller).

# Using pre-trained checkpoints

Pretrained checkpoints for 30 and 150 particle datasets are provided for both the initial FPCD model (using 512 steps during generation) and the distilled model for single-shot generation. Those can be directly sampled using the commands

```bash
python plot_jet.py [--distill --factor 512] [--big] --sample
```


