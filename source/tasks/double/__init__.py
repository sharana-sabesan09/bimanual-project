import gymnasium as gym

gym.register(
    id="BallBalance-DualArm-v0",
    entry_point="source.tasks.double.env:DualArmBallBalanceEnv",
    disable_env_checker=True,
    kwargs={
        "rsl_rl_cfg_entry_point": "source.tasks.double.agents.rsl_rl_ppo_cfg:get_train_cfg",
    },
)

# Same env, custom LSTM actor + deep MLP critic — swap just by changing --task
gym.register(
    id="BallBalance-DualArm-LSTM-v0",
    entry_point="source.tasks.double.env:DualArmBallBalanceEnv",
    disable_env_checker=True,
    kwargs={
        "rsl_rl_cfg_entry_point": "source.tasks.double.agents.rsl_rl_ppo_cfg_custom:get_train_cfg",
    },
)
