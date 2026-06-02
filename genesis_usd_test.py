"""
Minimal proof-of-concept: load a USD robot asset into Genesis and step with zero actions.
"""

import torch
import genesis as gs
from pathlib import Path

USD_PATH = str(Path(__file__).parent / "assets" / "g1-flattened-fixed.usd")
N_STEPS = 10000

gs.init(backend=gs.cpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.5, -2.5, 2.0),
        camera_lookat=(0.0, 0.0, 1.0),
        camera_fov=45,
    ),
    show_viewer=True,
    sim_options=gs.options.SimOptions(dt=0.02),
)

scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.USD(file=USD_PATH, pos=(0.0, 0.0, 0.79)))

scene.build(n_envs=1)

pos = robot.get_pos()
pos[:, 2] += 1.2
robot.set_pos(pos)
n_dofs = robot.n_dofs
print(f"Loaded USD robot — DOFs: {n_dofs}")

# Cache the initial base pose so we can re-pin it every step
base_pos  = robot.get_pos()   # (1, 3)
base_quat = robot.get_quat()  # (1, 4)

# shape: (n_envs=1, n_dofs)
zero_pos = torch.rand(1, n_dofs, dtype=torch.float32, device=gs.device)

for step in range(N_STEPS):
    robot.control_dofs_position(zero_pos)
    scene.step()
    # Re-pin base after physics so gravity never accumulates on the root link
    # robot.set_pos(base_pos, zero_velocity=False)
    # robot.set_quat(base_quat, zero_velocity=False)
    if step % 50 == 0:
        pos = robot.get_dofs_position()
        print(f"  step {step:4d} | joint pos (first 6): {pos[0, :6].tolist()}")

print("Done.")
