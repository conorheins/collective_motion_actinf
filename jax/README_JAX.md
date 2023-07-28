# JAX set up instructions

## Set up / installation

To set up the JAX codebase, we recommend the following steps:

1. Creating a virtual environment using [`venv`](https://docs.python.org/3/library/venv.html) or [`conda`](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment). This environment will have all the required dependencies (correct versions of numpy, JAX, etc.).
2. Activate your virtual environment then install the latest version of `JAX` using `pip`. Please see the official instructions [here](https://github.com/google/jax#installation). Note that you can optionally install the CPU version or GPU versions of JAX, but be mindful of the capabilities of the machine you're running the code on. For the GPU-enabled version of JAX, you need to have an NVIDIA/CUDA-compatible GPU.
3. After installing JAX, please install the remaining requirements in your virtual environment using `pip install -r requirements.txt`

## Running the code 

Navigate into the `jax` directory from the root directory of the repository (e.g., `cd jax/`) and run a demo script using e.g.,

```
python src/demo_nolearning.py --<keyword1> <val1> --<keyword2> <val21> .... # optional number of keyword/parameter pairs
```

For example to run a 10-agent simulation for 20 seconds, and show the results of the last 10 seconds of all agents' trajectories, try something like:
```
python src/demo_nolearning.py -s 2 -N 10 -dt 0.01 -T 20 -lastT 10
``` 
You can also append the optional argument `--save` to a `demo` script, which will save the history of positions and velocities locally in an `.npz` file. The `-s` argument specifies the random seed used by JAX (an integer); this determines the values of any sampled/stochastic components in the simulation (e.g., the initial positions and velocities of the agents, realizations of observation noise, the realizations of action noise). This can thus be used to control reproducibility.



