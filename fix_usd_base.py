"""
Adds a world→pelvis PhysicsFixedJoint to g1-flattened.usd so the base is fixed.
Writes a new file: assets/g1-flattened-fixed.usd

Run with: ./isaaclab.sh -p fix_usd_base.py   (from IsaacLab root)
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics, Sdf

IN_PATH  = "assets/g1-flattened.usd"
OUT_PATH = "assets/g1-flattened-fixed.usd"
PELVIS   = "/g1_29dof_with_hand_rev_1_0/pelvis"

stage = Usd.Stage.Open(IN_PATH)

# Add a fixed joint under the pelvis prim that welds it to the world
joint_path = PELVIS + "/world_fixed_joint"
joint_prim = UsdPhysics.FixedJoint.Define(stage, joint_path)

# body0 left empty = world frame; body1 = pelvis
joint_prim.GetBody1Rel().SetTargets([Sdf.Path(PELVIS)])

stage.Export(OUT_PATH)
print(f"Written: {OUT_PATH}")
print(f"Joint added at: {joint_path}")
print(f"  body0 = world  |  body1 = {PELVIS}")

simulation_app.close()
