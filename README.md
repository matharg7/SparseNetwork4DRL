<p align="center">

<h2 align="center">Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning (ICML'25)</a></h2>
  <p align="center">
    <a href="https://guozheng-ma.github.io/"><strong>Guozheng Ma*</strong></a>
    ·
    <a href="https://github.com/lilucse"><strong>Lu Li*</strong></a>
    ·
    <a href="https://zerlinwang.github.io/"><strong>Zilin Wang</strong></a>
  </p>
  <p align="center">
    <a href="https://sites.google.com/site/mathshenli/home"><strong>Li Shen</strong></a>
    ·
    <a href="https://pierrelucbacon.com/"><strong>Pierre-Luc Bacon</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=en"><strong>DaCheng Tao</strong></a>
  </p>



This repository contains the source code required to reproduce the DeepMind Control experiments presented in our paper.

## 📋Getting started


### python environment

```
conda env create -f deps/environment.yaml
```

#### (optional) Jax for GPU
```
pip install -U "jax[cuda12]==0.4.30" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# If you want to execute multiple runs with a single GPU, we recommend to set this variable.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```




##  🚀Example usage

Below is an example of how to train a SAC agent using the SimBa network with a sparsity level of 0.8 on the humanoid-run environment.

```
python run.py \
        --config_name base_sac \
        --overrides seed=0 \
        --overrides updates_per_interaction_step=2 \
        --overrides actor_sparsity=0.8 \
        --overrides actor_num_blocks=1 \
        --overrides actor_hidden_dim=128 \
        --overrides critic_sparsity=0.8 \
        --overrides critic_num_blocks=2 \
        --overrides critic_hidden_dim=512 \
        --overrides env_name=humanoid-run
```

## 🙏 Acknowledgements

We would like to thank the [SimBa codebase](https://github.com/SonyResearch/simba) and [JaxPruner](https://github.com/google-research/jaxpruner). Our implementation builds on top of their repository.

## ❓Questions

If you have any questions, please raise an issue or send an email to Lu (lu.li@mila.quebec) and Guozheng (guozheng001@e.ntu.edu.sg).





