from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp

from scale_rl.buffers import Batch
from scale_rl.networks.trainer import PRNGKey, Trainer
from scale_rl.networks.utils import tree_norm
from scale_rl.networks.metrics import flatten_dict,get_dormant_ratio,get_feature_norm,get_weight_norm,get_srank
from scale_rl.agents.ddpg.ddpg_network import DDPGCritic
 
def update_actor(
    key: PRNGKey,
    actor: Trainer,
    critic: Trainer,  # SACDoubleCritic
    batch: Batch,
    critic_use_cdq: bool,
    noise_std: float,
) -> Tuple[Trainer, Dict[str, float]]:
    def actor_loss_fn(
        actor_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        actions = actor.apply(
            variables={"params": actor_params},
            observations=batch["observation"],
        )
        noise = noise_std * jax.random.normal(key, shape=actions.shape)
        actions = jnp.clip(actions + noise, -1.0, 1.0)

        if critic_use_cdq:
            q1, q2 = critic(observations=batch["observation"], actions=actions)
            q = jnp.minimum(q1, q2).reshape(-1)  # (n, 1) -> (n, )
        else:
            q = critic(observations=batch["observation"], actions=actions)
            q = q.reshape(-1)

        actor_loss = -q.mean()
        actor_info = {
            "train/actor_loss": actor_loss,
            "train/actor_action": jnp.mean(jnp.abs(actions)),
            "train/actor_pnorm": tree_norm(actor_params),
        }

        return actor_loss, actor_info

    actor, info = actor.apply_gradient(actor_loss_fn)
    info["train/actor_gnorm"] = info.pop("grad_norm")

    return actor, info


def get_actor_with_metrics(
    actor: Trainer,
    batch: Batch,
):
    actions,intermediates  = actor.apply(
        variables={"params": actor.params},
        observations=batch["observation"],
        capture_intermediates=True,
        )
    newintermediates = flatten_dict(intermediates)
    actor_DR1 = get_dormant_ratio(newintermediates,prefix='actor',tau=0.1)
    actor_DR2 = get_dormant_ratio(newintermediates,prefix='actor',tau=0.2)
    actor_FN = get_feature_norm(newintermediates['intermediates_encoder___call__'])
    actor_WN = get_weight_norm(actor.params,prefix='actor')
    actor_srank = get_srank(newintermediates['intermediates_encoder___call__'],thershold=0.01)
    actor_info = {
        "train/actor_DR0.1": actor_DR1["actor/dormant_total"],
        "train/actor_DR0.2": actor_DR2["actor/dormant_total"],
        "train/actor_fnorm": actor_FN,
        "train/actor_wnorm": actor_WN['actor/weightnorm_total'],
        "train/actor_srank": actor_srank,
    }
    return actor_info
    

def update_critic(
    key: PRNGKey,
    actor: Trainer,
    critic: Trainer,
    target_critic: Trainer,
    batch: Batch,
    gamma: float,
    n_step: int,
    critic_use_cdq: bool,
    noise_std: float,
) -> Tuple[Trainer, Dict[str, float]]:
    # compute the target q-value
    next_actions = actor(observations=batch["next_observation"])
    noise = noise_std * jax.random.normal(key, shape=next_actions.shape)
    next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

    if critic_use_cdq:
        next_q1, next_q2 = target_critic(
            observations=batch["next_observation"], actions=next_actions
        )
        next_q = jnp.minimum(next_q1, next_q2).reshape(-1)
    else:
        next_q = target_critic(
            observations=batch["next_observation"],
            actions=next_actions,
        ).reshape(-1)

    # compute the td-target, incorporating the n-step accumulated reward
    # https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
    target_q = batch["reward"] + (gamma**n_step) * (1 - batch["terminated"]) * next_q

    def critic_loss_fn(
        critic_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        # compute predicted q-value
        if critic_use_cdq:
            pred_q1, pred_q2 = critic.apply(
                variables={"params": critic_params},
                observations=batch["observation"],
                actions=batch["action"],
            )
            pred_q1 = pred_q1.reshape(-1)
            pred_q2 = pred_q2.reshape(-1)

            # compute mse loss
            critic_loss = ((pred_q1 - target_q) ** 2 + (pred_q2 - target_q) ** 2).mean()
        else:
            pred_q = critic.apply(
                variables={"params": critic_params},
                observations=batch["observation"],
                actions=batch["action"],
            ).reshape(-1)
            pred_q1 = pred_q2 = pred_q

            # compute mse loss
            critic_loss = ((pred_q - target_q) ** 2).mean()

        critic_info = {
            "train/critic_loss": critic_loss,
            "train/q1_mean": pred_q1.mean(),
            "train/q2_mean": pred_q2.mean(),
            "train/rew_mean": batch["reward"].mean(),
            "train/critic_pnorm": tree_norm(critic_params),
        }

        return critic_loss, critic_info

    critic, info = critic.apply_gradient(critic_loss_fn)
    info["train/critic_gnorm"] = info.pop("grad_norm")

    return critic, info

def get_critic_with_metrics(
    key: PRNGKey, 
    actor: Trainer,
    critic: Trainer,
    batch: Batch,
    critic_use_cdq,
):
    if critic_use_cdq:
        intermediates= critic.apply(
                variables={"params": critic.params},
                observations=batch["observation"],
                actions=batch["action"],
                capture_intermediates=True,
            )[1]
    else:
        q, intermediates= critic.apply(
                variables={"params": critic.params},
                observations=batch["observation"],
                actions=batch["action"],
                capture_intermediates=True,
            )

    newintermediates = flatten_dict(intermediates)

    if critic_use_cdq:
        q1_intermediates = {}
        q2_intermediates = {}
        for layer_name, activi in list(newintermediates.items()):
            jnp_activi_q1 = jnp.array(activi)[:,0,:,:]
            jnp_activi_q2 = jnp.array(activi)[:,1,:,:]
            q1_intermediates.update({layer_name:jnp_activi_q1})
            q2_intermediates.update({layer_name:jnp_activi_q2})
    
        critic_q1_DR1 = get_dormant_ratio(q1_intermediates,prefix='critic',tau=0.1)
        critic_q1_DR2 = get_dormant_ratio(q1_intermediates,prefix='critic',tau=0.2)

        critic_q2_DR1 = get_dormant_ratio(q2_intermediates,prefix='critic',tau=0.1)
        critic_q2_DR2 = get_dormant_ratio(q2_intermediates,prefix='critic',tau=0.2)

        critic_q1_FN = get_feature_norm(q1_intermediates['intermediates_VmapDDPGCritic_0_encoder___call__'])
        critic_q2_FN = get_feature_norm(q2_intermediates['intermediates_VmapDDPGCritic_0_encoder___call__'])
        critic_q1_srank = get_srank(q1_intermediates['intermediates_VmapDDPGCritic_0_encoder___call__'],thershold=0.01)
        critic_q2_srank = get_srank(q2_intermediates['intermediates_VmapDDPGCritic_0_encoder___call__'],thershold=0.01)

        critic_WN = get_weight_norm(critic.params,prefix='critic')
        critic_info = {
            "train/critic_q1_DR0.1": critic_q1_DR1["critic/dormant_total"],
            "train/critic_q1_DR0.2": critic_q1_DR2["critic/dormant_total"],
            "train/critic_q2_DR0.1": critic_q2_DR1["critic/dormant_total"],
            "train/critic_q2_DR0.2": critic_q2_DR2["critic/dormant_total"],
            "train/critic_q1_fnorm": critic_q1_FN,
            "train/critic_q2_fnorm": critic_q2_FN,
            "train/critic_wnorm": critic_WN['critic/weightnorm_total'],
            "train/critic_q1_srank": critic_q1_srank,
            "train/critic_q2_srank": critic_q2_srank,
        }
    else:
        critic_DR1 = get_dormant_ratio(newintermediates,prefix='critic',tau=0.1)
        critic_DR2 = get_dormant_ratio(newintermediates,prefix='critic',tau=0.2)
        critic_FN = get_feature_norm(newintermediates['intermediates_encoder___call__'])
        critic_srank = get_srank(newintermediates['intermediates_encoder___call__'],thershold=0.01)
        critic_WN = get_weight_norm(critic.params,prefix='critic')
        critic_info = {
            "train/critic_DR0.1": critic_DR1["critic/dormant_total"],
            "train/critic_DR0.2": critic_DR2["critic/dormant_total"],
            "train/critic_fnorm": critic_FN,
            "train/critic_wnorm": critic_WN['critic/weightnorm_total'],
            "train/critic_srank": critic_srank,
        }
    return critic_info

def update_target_network(
    network: Trainer,
    target_network: Trainer,
    target_tau: float,
) -> Tuple[Trainer, Dict[str, float]]:
    new_target_params = jax.tree.map(
        lambda p, tp: p * target_tau + tp * (1 - target_tau),
        network.params,
        target_network.params,
    )

    target_network = target_network.replace(params=new_target_params)
    info = {}

    return target_network, info

