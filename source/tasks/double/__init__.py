import gymnasium as gym

gym.register(
    id="BallBalance-DualArm-v0",
    entry_point="source.tasks.double.env:DualArmBallBalanceEnv",
    disable_env_checker=True,
    kwargs={
        "rsl_rl_cfg_entry_point": "source.tasks.double.agents.rsl_rl_ppo_cfg:get_train_cfg",
    },
)
