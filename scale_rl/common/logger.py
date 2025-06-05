from typing import Dict

from omegaconf import OmegaConf

import wandb

def get_run_info(args):
    if args.agent.actor_sparsity == 0.0 and args.agent.actor_sparsity == 0.0:
        group = f"Dense_{args.agent.agent_type}_{args.env.env_name}_RR={args.updates_per_interaction_step}_arch={args.agent.critic_block_type[:3]}"
    else:
        group = f"Sparse_{args.agent.agent_type}_{args.env.env_name}_RR={args.updates_per_interaction_step}_arch={args.agent.critic_block_type[:3]}"
    if args.agent.critic_block_type == 'residual':
        job_type = f"P={args.num_params[:-1]}_AP={args.actor_num_params[:-1]}_D={args.agent.actor_num_blocks}_W={args.agent.actor_hidden_dim}_S={args.agent.actor_sparsity}_CP={args.critic_num_params[:-1]}_D={args.agent.critic_num_blocks}_W={args.agent.critic_hidden_dim}_S={args.agent.critic_sparsity}"
        name = f"seed={args.agent.seed}_{args.agent.agent_type}_{args.env.env_name}_RR={args.updates_per_interaction_step}_P={args.num_params}_AP={args.actor_num_params}_AD={args.agent.actor_num_blocks}_AW={args.agent.actor_hidden_dim}_AS={args.agent.actor_sparsity}_CP={args.critic_num_params}_CD={args.agent.critic_num_blocks}_CW={args.agent.critic_hidden_dim}_CS={args.agent.critic_sparsity}"
    else:
        job_type = f"P={args.num_params}_AP={args.actor_num_params}D={args.agent.actor_num_blocks}W={args.agent.actor_hidden_dim}S={args.agent.actor_sparsity}CP={args.critic_num_params}D={args.agent.critic_num_blocks}W={args.agent.critic_hidden_dim}S={args.agent.critic_sparsity}"
        name = f"seed={args.agent.seed}_{args.agent.agent_type}_{args.env.env_name}_RR={args.updates_per_interaction_step}_P={args.num_params}_AP={args.actor_num_params}_AD={args.agent.actor_num_blocks}_AW={args.agent.actor_hidden_dim}_AS={args.agent.actor_sparsity}_CP={args.critic_num_params}_CD={args.agent.critic_num_blocks}_CW={args.agent.critic_hidden_dim}_CS={args.agent.critic_sparsity}"
    return {
        "group": group,
        "job_type": job_type,
        "name": name
    }
class WandbTrainerLogger(object):
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        dict_cfg = OmegaConf.to_container(cfg, throw_on_missing=True)
        run_info = get_run_info(cfg)

        wandb.init(
            project=cfg.project_name,
            group=run_info["group"],
            config=dict_cfg,
            job_type=run_info["job_type"],
            name=run_info["name"], 
        )

        self.reset()

    def update_metric(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, float) or isinstance(v, int):
                self.average_meter_dict.update(k, v)
            else:
                self.media_dict[k] = v

    def log_metric(self, step: int) -> Dict:
        log_data = {}
        log_data.update(self.average_meter_dict.averages())
        log_data.update(self.media_dict)
        wandb.log(log_data, step=step)

    def reset(self) -> None:
        self.average_meter_dict = AverageMeterDict()
        self.media_dict = {}


class AverageMeterDict(object):
    """
    Manages a collection of AverageMeter instances,
    allowing for grouped tracking and averaging of multiple metrics.
    """

    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string="{}"):
        return {
            format_string.format(name): meter.val for name, meter in self.meters.items()
        }

    def averages(self, format_string="{}"):
        return {
            format_string.format(name): meter.avg for name, meter in self.meters.items()
        }


class AverageMeter(object):
    """
    Tracks and calculates the average and current values of a series of numbers.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # TODO: description for using n
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )
