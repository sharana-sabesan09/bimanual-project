"""
Custom architecture config for the dual-arm ball-balance task.
Swaps in LSTMActor + DeepMLPCritic from RL_lib/ — all other hyperparameters
are identical to rsl_rl_ppo_cfg.py.

Usage in train.py:
    from source.tasks.double.agents.rsl_rl_ppo_cfg_custom import get_train_cfg
"""


def get_train_cfg(exp_name: str) -> dict:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.001,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 5e-5,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            "rnd_cfg": None,
        },
        "actor": {
            "class_name": "RL_lib.custom_actor_critic.LSTMActor", # example of adjusting class name to our custom nn
            "hidden_size": 256,
            "num_layers": 1,
            "hidden_dims": [256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "RL_lib.custom_actor_critic.DeepMLPCritic", # # example of adjusting class name to our custom nn
            "hidden_dims": [512, 512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 128,
        "save_interval": 25,
        "run_name": exp_name,
        "logger": "tensorboard",
    }
