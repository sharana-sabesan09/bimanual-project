"""
Quick benchmark of performance_mode=True vs False.
Must run in a fresh process (gs.init can only be called once per process).
"""
import time, sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--perf_mode", action="store_true")
args = parser.parse_args()

import genesis as gs
from pathlib import Path
G1_USD = str(Path(__file__).parents[1] / "assets" / "g1-flattened-fixed.usd")

N = 32
WARMUP, ITERS = 20, 150

gs.init(backend=gs.gpu, logging_level="error",
        **({"performance_mode": True} if args.perf_mode else {}))

scene = gs.Scene(show_viewer=False,
                 sim_options=gs.options.SimOptions(dt=0.02, substeps=4))
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.USD(file=G1_USD, pos=(0.0, 0.0, 1.0)))
scene.add_entity(gs.morphs.Box(size=(0.36, 0.26, 0.01)), material=gs.materials.Rigid(rho=300.0))
scene.add_entity(gs.morphs.Sphere(radius=0.03, pos=(0.0, 0.0, 1.5)), material=gs.materials.Rigid(rho=8842.0))
scene.build(n_envs=N, env_spacing=(1.5, 1.5))

for _ in range(WARMUP):
    scene.step()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(ITERS):
    scene.step()
torch.cuda.synchronize()
t1 = time.perf_counter()
ms = (t1 - t0) / ITERS * 1000
mode = "performance_mode=True" if args.perf_mode else "default"
print(f"{mode}: {ms:.3f} ms/step  ({N / (ms/1000):,.0f} env-steps/s)")
