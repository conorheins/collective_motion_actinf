# Collective motion through multi-agent active inference
This repository provides code for simulating emergent collective motion from groups of continuous-time and -space active inference agents. This code serves as companion for the preprint ["Collective behavior from surprise minimization"](https://arxiv.org/abs/2307.14804) (2023) by Conor Heins, Beren Millidge, Lancelot Da Costa, Richard Mann, Karl Friston, and Iain Couzin.

This codebase contains both [`JAX`](https://github.com/google/jax) and a [`Julia`](https://julialang.org/) implementations of a multi-agent active inference algorithm for generating collective motion. The `JAX` implementation (in the `jax` folder) is the recommended implementation, especially because all code automatically takes advantage of GPU-support on machines with NVIDIA-capable GPUs (see [here](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier) for the official instructions). 

- `JAX` installation/run instructions [here](https://github.com/conorheins/collective_motion_actinf/blob/main/jax/README_JAX.md)
- `Julia` installation/run instructions [here](https://github.com/conorheins/collective_motion_actinf/blob/main/julia/README_Julia.md)

