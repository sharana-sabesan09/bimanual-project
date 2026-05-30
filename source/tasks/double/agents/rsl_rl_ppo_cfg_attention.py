def get_train_cfg(exp_name: str) -> dict:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.015,
            "entropy_coef": 3e-4,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 8,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            "rnd_cfg": None,
        },

        "actor": {
            "class_name": "RL_lib.attention_actor_critic.AttentionActor",
            "hidden_dims": [256, 128],
            "num_heads": 4,
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },

        "critic": {
            "class_name": "RL_lib.attention_actor_critic.AttentionCritic",
            "hidden_dims": [512, 256, 128],
            "num_heads": 4,
            "activation": "elu",
        },

        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },

        "num_steps_per_env": 128,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }