import functools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import dynamic_scale
import ml_collections
import jaxpruner

from scale_rl.agents.sparse_training_api import create_sparse_updater

from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.sac.sac_network import (
    SACActor,
    SACClippedDoubleCritic,
    SACCritic,
    SACTemperature,
)
from scale_rl.agents.sac.sac_update import (
    update_actor,
    update_critic,
    update_target_network,
    update_temperature,
    get_actor_with_metrics,
    get_critic_with_metrics,
    compute_actor_gradient_cosine,
)
from scale_rl.buffers.base_buffer import Batch
from scale_rl.networks.trainer import PRNGKey, Trainer
from scale_rl.agents.sparse import get_sparsities_erdos_renyi,get_var_shape_dict,create_mask
from scale_rl.networks.metrics import print_num_parameters,flatten_dict,format_params_str
"""
The @dataclass decorator must have `frozen=True` to ensure the instance is immutable,
allowing it to be treated as a static variable in JAX.
"""


@dataclass(frozen=True)
class SACConfig:
    seed: int
    num_train_envs: int
    max_episode_steps: int
    normalize_observation: bool

    actor_block_type: str
    actor_num_blocks: int
    actor_hidden_dim: int
    actor_learning_rate: float
    actor_weight_decay: float

    critic_block_type: str
    critic_num_blocks: int
    critic_hidden_dim: int
    critic_learning_rate: float
    critic_weight_decay: float
    critic_use_cdq: bool

    temp_target_entropy: float
    temp_target_entropy_coef: float
    temp_initial_value: float
    temp_learning_rate: float
    temp_weight_decay: float

    target_tau: float
    gamma: float
    n_step: int

    mixed_precision: bool

    actor_pruner: str
    actor_sparsity: float
    actor_update_frequency: int
    actor_start_step: int
    actor_end_step: int
    actor_sparsity_distribution: str
    actor_drop_fraction: float

    critic_pruner: str
    critic_sparsity: float
    critic_update_frequency: int
    critic_start_step: int
    critic_end_step: int
    critic_sparsity_distribution: str
    critic_drop_fraction: float


# @functools.partial(
#     jax.jit,
#     static_argnames=(
#         "observation_dim",
#         "action_dim",
#         "cfg",
#     ),
# )

def create_jaxpruner_configs(config):
    actor_jaxpruner_config = ml_collections.ConfigDict()
    actor_jaxpruner_config.algorithm = config.actor_pruner
    actor_jaxpruner_config.sparsity = config.actor_sparsity
    actor_jaxpruner_config.update_freq = config.actor_update_frequency
    actor_jaxpruner_config.update_start_step = config.actor_start_step
    actor_jaxpruner_config.update_end_step = config.actor_end_step
    actor_jaxpruner_config.dist_type = config.actor_sparsity_distribution
    actor_jaxpruner_config.drop_fraction = config.actor_drop_fraction

    critic_jaxpruner_config = ml_collections.ConfigDict()
    critic_jaxpruner_config.algorithm = config.critic_pruner
    critic_jaxpruner_config.sparsity = config.critic_sparsity
    critic_jaxpruner_config.update_freq = config.critic_update_frequency
    critic_jaxpruner_config.update_start_step = config.critic_start_step
    critic_jaxpruner_config.update_end_step = config.critic_end_step
    critic_jaxpruner_config.dist_type = config.critic_sparsity_distribution
    critic_jaxpruner_config.drop_fraction = config.critic_drop_fraction
    
    dense = ml_collections.ConfigDict()
    dense.algorithm = "no_prune"

    return actor_jaxpruner_config, critic_jaxpruner_config, dense


def _init_sac_networks(
    observation_dim: int,
    action_dim: int,
    cfg: SACConfig,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer]:
    fake_observations = jnp.zeros((1, observation_dim))
    fake_actions = jnp.zeros((1, action_dim))

    rng = jax.random.PRNGKey(cfg.seed)
    rng, actor_key, critic_key, temp_key, actor_rng_sparse, critic_rng_sparse = jax.random.split(rng, 6)
    compute_dtype = jnp.float16 if cfg.mixed_precision else jnp.float32

    # if cfg.actor_sparsity > 0.0:
    #     actor_sparse = True
    # else:
    #    actor_sparse = False
    # When initializing the network in the flax.nn.Module class, rng_key should be passed as rngs.
    
    # if actor_sparse:
    actor_sparse_config, critic_sparse_config, dense = create_jaxpruner_configs(cfg)
    
    
    actor_sparse_config.rng_seed = actor_rng_sparse
    actor_pruner = create_sparse_updater(actor_sparse_config)
    
    critic_sparse_config.rng_seed = critic_rng_sparse
    critic_pruner = create_sparse_updater(critic_sparse_config)

    dense_pruner = create_sparse_updater(dense)

    actor_network_def=SACActor(
            block_type=cfg.actor_block_type,
            num_blocks=cfg.actor_num_blocks,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=action_dim,
            dtype=compute_dtype,
        )
    # _fake_actor_variables = actor_network_def.init(rng, fake_observations)
    # fake_actor_params = _fake_actor_variables['params']  # A FrozenDict of parameters
    # if actor_sparse:
    #     actor_network_parm_dict = get_var_shape_dict(fake_actor_params)
    #     actor_network_sparse = get_sparsities_erdos_renyi(actor_network_parm_dict, default_sparsity=float(cfg.actor_sparsity))
    #     mask_rng = jax.random.PRNGKey(cfg.seed)
    #     actor_network_mask = create_mask(fake_actor_params, actor_network_sparse, mask_rng)
    # else:
    #     actor_network_mask = None

    actor_optimizer = optax.adamw(
            learning_rate=cfg.actor_learning_rate,
            weight_decay=cfg.actor_weight_decay,
        )

    # actor_optimizer = actor_pruner.wrap_optimizer(actor_optimizer)

    actor = Trainer.create(
        network_def=actor_network_def,
        network_inputs={"rngs": actor_key, "observations": fake_observations},
        tx=actor_optimizer,
        dynamic_scale=dynamic_scale.DynamicScale() if cfg.mixed_precision else None,
        pruner=actor_pruner
        # sparse = actor_sparse,
        # network_mask = actor_network_mask
    )


    if cfg.critic_use_cdq:
        critic_network_def = SACClippedDoubleCritic(
            block_type=cfg.critic_block_type,
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=compute_dtype,
        )
    else:
        critic_network_def = SACCritic(
            block_type=cfg.critic_block_type,
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            dtype=compute_dtype,
        )
    # _fake_critic_variables = critic_network_def.init(rng, observations=fake_observations, actions=fake_actions)
    # fake_critic_params = _fake_critic_variables['params']  # A FrozenDict of parameters
    # if cfg.actor_sparsity > 0.0:
    #     critic_sparse = True
    # else:
    #    critic_sparse = False
    # if critic_sparse:
    #     critic_network_parm_dict = get_var_shape_dict(fake_critic_params)
    #     critic_network_sparse = get_sparsities_erdos_renyi(critic_network_parm_dict, default_sparsity=float(cfg.critic_sparsity))
    #     mask_rng = jax.random.PRNGKey(cfg.seed)
    #     critic_network_mask = create_mask(fake_critic_params, critic_network_sparse, mask_rng)
    # else:
    #     critic_network_mask = None
    critic_optimizer = optax.adamw(
            learning_rate=cfg.critic_learning_rate,
            weight_decay=cfg.critic_weight_decay,
        )

    # critic_optimizer = critic_pruner.wrap_optimizer(critic_optimizer)
    
    critic = Trainer.create(
        network_def=critic_network_def,
        network_inputs={
            "rngs": critic_key,
            "observations": fake_observations,
            "actions": fake_actions,
        },
        tx=critic_optimizer,
        dynamic_scale=dynamic_scale.DynamicScale() if cfg.mixed_precision else None,
        pruner=critic_pruner,
        # sparse = critic_sparse,
        # network_mask = critic_network_mask,
    )

    # we set target critic's parameters identical to critic by using same rng.
    target_network_def = critic_network_def
    target_critic = Trainer.create(
        network_def=target_network_def,
        network_inputs={
            "rngs": critic_key,
            "observations": fake_observations,
            "actions": fake_actions,
        },
        tx=None,
        pruner=dense_pruner
        # sparse = critic_sparse,
        # network_mask = critic_network_mask,
    )

    temperature = Trainer.create(
        network_def=SACTemperature(cfg.temp_initial_value),
        network_inputs={
            "rngs": temp_key,
        },
        tx=optax.adamw(
            learning_rate=cfg.temp_learning_rate,
            weight_decay=cfg.temp_weight_decay,
        ),
        pruner=dense_pruner
        # sparse=False,
        # network_mask=None,
    )

    return rng, actor, critic, target_critic, temperature


@jax.jit
def _sample_sac_actions(
    rng: PRNGKey,
    actor: Trainer,
    observations: jnp.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    rng, key = jax.random.split(rng)
    dist = actor(observations=observations, temperature=temperature)
    actions = dist.sample(seed=key)

    return rng, actions


@functools.partial(
    jax.jit,
    static_argnames=(
        "gamma",
        "n_step",
        "critic_use_cdq",
        "target_tau",
        "temp_target_entropy",
    ),
)
def _update_sac_networks(
    rng: PRNGKey,
    actor: Trainer,
    critic: Trainer,
    target_critic: Trainer,
    temperature: Trainer,
    batch: Batch,
    gamma: float,
    n_step: int,
    critic_use_cdq: bool,
    target_tau: float,
    temp_target_entropy: float,
) -> Tuple[PRNGKey, Trainer, Trainer, Trainer, Trainer, Dict[str, float]]:
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    new_actor, actor_info = update_actor(
        key=actor_key,
        actor=actor,
        critic=critic,
        temperature=temperature,
        batch=batch,
        critic_use_cdq=critic_use_cdq,
    )

    new_temperature, temperature_info = update_temperature(
        temperature=temperature,
        entropy=actor_info["train/entropy"],
        target_entropy=temp_target_entropy,
    )

    new_critic, critic_info = update_critic(
        key=critic_key,
        actor=new_actor,
        critic=critic,
        target_critic=target_critic,
        temperature=new_temperature,
        batch=batch,
        gamma=gamma,
        n_step=n_step,
        critic_use_cdq=critic_use_cdq,
    )

    new_target_critic, target_critic_info = update_target_network(
        network=new_critic,
        target_network=target_critic,
        target_tau=target_tau,
    )

    info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
    }

    return (rng, new_actor, new_critic, new_target_critic, new_temperature, info)


class SACAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: SACConfig,
    ):
        """
        An agent that randomly selects actions without training.
        Useful for collecting baseline results and for debugging purposes.
        """

        self._observation_dim = observation_space.shape[-1]
        self._action_dim = action_space.shape[-1]

        cfg["temp_target_entropy"] = cfg["temp_target_entropy_coef"] * self._action_dim

        super(SACAgent, self).__init__(
            observation_space,
            action_space,
            cfg,
        )
        print(cfg.items())
        # map dictionary to dataclass
        self._cfg = SACConfig(**cfg)

        self._init_network()

    def _init_network(self):
        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
        ) = _init_sac_networks(self._observation_dim, self._action_dim, self._cfg)

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        if training:
            temperature = 1.0
        else:
            temperature = 0.0

        # current timestep observation is "next" observations from the previous timestep
        observations = jnp.asarray(prev_timestep["next_observation"])

        self._rng, actions = _sample_sac_actions(
            self._rng, self._actor, observations, temperature
        )
        actions = np.array(actions)

        return actions

    def update(self, update_step: int, batch: Dict[str, np.ndarray]) -> Dict:
        for key, value in batch.items():
            batch[key] = jnp.asarray(value)

        (
            self._rng,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
            update_info,
        ) = _update_sac_networks(
            rng=self._rng,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            batch=batch,
            gamma=self._cfg.gamma,
            n_step=self._cfg.n_step,
            critic_use_cdq=self._cfg.critic_use_cdq,
            target_tau=self._cfg.target_tau,
            temp_target_entropy=self._cfg.temp_target_entropy,
        )

        for key, value in update_info.items():
            update_info[key] = float(value)

        return update_info

    def get_metrics(self, update_step: int, batch: Dict[str, np.ndarray]):
        for key, value in batch.items():
            batch[key] = jnp.asarray(value)
        actor_metrics_info = get_actor_with_metrics(actor=self._actor,batch=batch)
        critic_metrics_info = get_critic_with_metrics(key=self._rng,actor=self._actor,critic=self._critic,batch=batch,critic_use_cdq=self._cfg.critic_use_cdq)
        combined_metric_info = {**actor_metrics_info, **critic_metrics_info}
        return combined_metric_info

    def get_num_parameters(self):
        actor_num_params = print_num_parameters(flatten_dict(self._actor.params),network_type='actor_simba')
        critic_num_params = print_num_parameters(flatten_dict(self._critic.params),network_type='critic_simba')
        total_params = actor_num_params + critic_num_params

        num_str = format_params_str(total_params)
        num_str_actor = format_params_str(actor_num_params)
        num_str_critic = format_params_str(critic_num_params)
        print(f" total params:",num_str)

        return  total_params, num_str, num_str_actor, num_str_critic