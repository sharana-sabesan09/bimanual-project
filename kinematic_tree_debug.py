import genesis as gs
from source.tasks.double.env import DualArmBallBalanceEnv

gs.init(backend=gs.cpu)
env = DualArmBallBalanceEnv(show_viewer=True, n_envs=1)
obs, _ = env.reset()

r_wrist = env.robot.get_link("right_wrist_yaw_link")
l_wrist = env.robot.get_link("left_wrist_yaw_link")
tray    = env.robot.get_link("tray")  # or env.scene.get_entity("tray")

print("right wrist pos:", r_wrist.get_pos())
print("right wrist quat:", r_wrist.get_quat())
print("left wrist pos:", l_wrist.get_pos())
print("left wrist quat:", l_wrist.get_quat())
print("tray pos:", tray.get_pos())
print("tray quat:", tray.get_quat())

while True:
    env.scene.step()
