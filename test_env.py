# test_env.py
#
# PURPOSE: Week 2 sanity check for the bimanual ball-balancing project.
# Before we write any RL training code, we need to confirm that:
#   1. Genesis can load the Unitree G1 robot from its MJCF file
#   2. Fixing the base works (robot stays upright, no locomotion needed)
#   3. A tray and ball can be added to the scene with physics
#   4. The simulation steps without crashing
#
# Nothing is learned here — this is purely environment validation.
# The tray and ball will fall since they are not yet attached to the robot hands.
# That gets handled in the next step (welding constraints or reward shaping).

import xml.etree.ElementTree as ET
import os
import genesis as gs

def make_fixed_base_mjcf(src_path):
    """
    Genesis's MJCF morph does not support a 'fixed' flag.
    The G1 XML has a <joint name="floating_base_joint" type="free"> on the pelvis
    which lets the whole robot fall freely. We remove it so the pelvis stays pinned,
    then write the result alongside the original file so that the relative
    meshdir="meshes" path still resolves correctly.
    """
    tree = ET.parse(src_path)
    root = tree.getroot()
    for body in root.iter("body"):
        for joint in body.findall("joint"):
            if joint.get("name") == "floating_base_joint":
                body.remove(joint)
                print("Removed floating_base_joint — robot base is now fixed.")
    # Write next to the original so relative mesh paths keep working
    out_path = os.path.join(os.path.dirname(src_path), "g1_29dof_fixed.xml")
    tree.write(out_path)
    return out_path

def main():
    # Initialize Genesis on CPU (Apple Silicon Mac has no CUDA).
    # On a Linux machine with an RTX GPU, swap gs.cpu -> gs.cuda for much faster sim.
    gs.init(backend=gs.cpu)

    # A Scene holds all physics objects and controls the simulation loop.
    # dt=0.01 means each step advances time by 10ms (100 Hz simulation).
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -2.0, 1.5),   # camera position in world space (x, y, z)
            camera_lookat=(0.0, 0.0, 0.8), # camera points at roughly the robot's torso
            camera_fov=40,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=True,  # opens an interactive window so we can see the scene
    )

    # Add a flat ground plane so objects don't fall into the void.
    scene.add_entity(gs.morphs.Plane())

    # Load the Unitree G1 humanoid robot from its MuJoCo XML (MJCF) file.
    # The G1 29-DOF version includes both arms (7 joints each) + legs + torso.
    # We strip the floating_base_joint first so the pelvis is pinned in place —
    # we are NOT doing locomotion, just bimanual arm control.
    fixed_mjcf = make_fixed_base_mjcf("unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
    robot = scene.add_entity(
        gs.morphs.MJCF(file=fixed_mjcf),
    )
    os.unlink(fixed_mjcf)  # clean up the generated file after Genesis has parsed it

    # Add the tray as a thin rigid box floating in front of the robot at hand height.
    # pos=(0.35, 0.0, 1.0) puts it ~35cm in front of the robot at 1m height.
    # In later steps this will be welded to both wrists or grasped via reward shaping.
    tray = scene.add_entity(
        gs.morphs.Box(
            size=(0.35, 0.25, 0.01),  # 35cm wide, 25cm deep, 1cm thick
            pos=(0.35, 0.0, 1.0),
            fixed=False,              # affected by gravity — will fall for now
        ),
        surface=gs.surfaces.Default(color=(0.6, 0.4, 0.2, 1.0)),  # brown
    )

    # Add the ball sitting on top of the tray.
    # pos z = tray z (1.0) + tray half-thickness (0.005) + ball radius (0.04) + tiny gap
    ball = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.04,
            pos=(0.35, 0.0, 1.06),
            fixed=False,  # affected by gravity and will roll — that's the task
        ),
        surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2, 1.0)),  # red
    )

    # Build compiles the scene — after this point no new entities can be added.
    scene.build()

    print("Scene built successfully.")
    print(f"Robot DOFs: {robot.n_dofs}")  # expect 29
    print(f"Robot joints: {[j.name for j in robot.joints]}")

    # Step the simulation indefinitely so the viewer stays open.
    # Press Ctrl+C in the terminal to stop.
    print("Simulation running — press Ctrl+C to stop.")
    while True:
        scene.step()

if __name__ == "__main__":
    main()
