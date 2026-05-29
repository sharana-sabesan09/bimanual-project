import gymnasium as gym

gym.register(
    id="BallBalance-SingleArm-v0",
    entry_point="source.tasks.single.env:SingleArmBallBalanceEnv",
    disable_env_checker=True,
    kwargs={
        "rsl_rl_cfg_entry_point": "source.tasks.single.agents.rsl_rl_ppo_cfg:get_train_cfg",
    },
)
