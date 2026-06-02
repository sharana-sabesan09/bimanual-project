import gymnasium as gym

gym.register(
    id="BallBalance-SingleArm-TrayGrasp-v0",
    entry_point="source.tasks.tray_grasp.single_arm_grasp:SingleArmTrayGraspEnv",
    disable_env_checker=True,
    kwargs={
        "rsl_rl_cfg_entry_point": "source.tasks.tray_grasp.agents.rsl_rl_ppo_cfg:get_train_cfg",
    },
)
