#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python run.py \
        --config_name base_sac \
        --overrides seed=0 \
        --overrides updates_per_interaction_step=2 \
        --overrides actor_pruner='static_sparse' \
        --overrides actor_sparsity=0.8 \
        --overrides actor_update_frequency=1500 \
        --overrides actor_start_step=0 \
        --overrides actor_end_step=625000 \
        --overrides actor_sparsity_distribution=erk \
        --overrides actor_drop_fraction=0.5 \
        --overrides actor_num_blocks=1 \
        --overrides actor_hidden_dim=128 \
        --overrides critic_pruner='static_sparse' \
        --overrides critic_sparsity=0.8 \
        --overrides critic_update_frequency=1500 \
        --overrides critic_start_step=0 \
        --overrides critic_end_step=625000 \
        --overrides critic_sparsity_distribution=erk \
        --overrides critic_drop_fraction=0.5 \
        --overrides critic_num_blocks=2 \
        --overrides critic_hidden_dim=512 \
        --overrides env_name=humanoid-run