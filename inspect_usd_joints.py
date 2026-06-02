"""
Inspect a USD file for ArticulationRoot, joints, and body hierarchy.
Run with: ./isaaclab.sh -p inspect_usd_joints.py   (from IsaacLab root)
  or:     ~/.local/share/ov/pkg/isaac-sim-*/python.sh inspect_usd_joints.py
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics, Gf

USD_PATH = "assets/g1-flattened.usd"

stage = Usd.Stage.Open(USD_PATH)

print("=" * 60)
print("ARTICULATION ROOTS")
print("=" * 60)
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print(f"  {prim.GetPath()}  [{prim.GetTypeName()}]")

print()
print("=" * 60)
print("ALL JOINTS  (type | path | body0 → body1)")
print("=" * 60)
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.Joint):
        continue
    j = UsdPhysics.Joint(prim)
    b0 = j.GetBody0Rel().GetTargets()
    b1 = j.GetBody1Rel().GetTargets()
    print(f"  [{prim.GetTypeName():20s}]  {prim.GetPath()}")
    print(f"      body0={b0}  body1={b1}")
    # Fixed joints
    if prim.IsA(UsdPhysics.FixedJoint):
        print("      -> FIXED")
    # D6 / prismatic hints at a free joint
    if prim.GetTypeName() in ("PhysicsJoint", "PhysicsPrismaticJoint", "PhysicsD6Joint"):
        # Check if all 6 axes are free
        d6 = UsdPhysics.DriveAPI
        print("      -> possible free/D6 joint")

print()
print("=" * 60)
print("TOP-LEVEL PRIMS (depth <= 2)")
print("=" * 60)
for prim in stage.GetPseudoRoot().GetChildren():
    print(f"  /{prim.GetName()}  [{prim.GetTypeName()}]")
    for child in prim.GetChildren():
        print(f"    /{prim.GetName()}/{child.GetName()}  [{child.GetTypeName()}]")

simulation_app.close()
