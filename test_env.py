# test_env.py

"""
just a sanity check before we write any actual training code
basically making sure genesis can load the g1 robot and that
the tray + ball show up in the scene without everything exploding

nothing is being learned here, the tray and ball will just fall
thats expected, we havent attached them to the hands yet
thats the next thing to figure out
"""

import xml.etree.ElementTree as ET
import os
import genesis as gs

def make_fixed_base_mjcf(src_path):
    # genesis doesnt have a "fixed base" option for mjcf files
    # the g1 xml has a free joint on the pelvis which makes the whole robot fall over
    # so we just delete that joint from the xml before loading it
    # we save the new xml next to the original so the mesh filepaths still work
    tree = ET.parse(src_path)
    root = tree.getroot()
    for body in root.iter("body"):
        for joint in body.findall("joint"):
            if joint.get("name") == "floating_base_joint":
                body.remove(joint)
                print("removed floating_base_joint, robot base is now fixed")
    out_path = os.path.join(os.path.dirname(src_path), "g1_29dof_fixed.xml")
    tree.write(out_path)
    return out_path

def main():
    # using cpu since were on mac, no cuda here
    # if we switch to a linux machine with a gpu swap this to gs.cuda
    gs.init(backend=gs.cpu)

    # dt=0.01 means each sim step = 10ms, so 100 steps = 1 second
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -2.0, 1.5),   # where the camera is in the world
            camera_lookat=(0.0, 0.0, 0.8), # roughly pointing at the robots torso
            camera_fov=40,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=True,  # opens the window so we can actually see whats happening
    )

    # floor so things dont fall forever
    scene.add_entity(gs.morphs.Plane())

    # load the g1 robot, 29 dof version has both arms (7 joints each) + legs + torso
    # we strip the free joint first so it stays standing in place
    fixed_mjcf = make_fixed_base_mjcf("unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
    robot = scene.add_entity(
        gs.morphs.MJCF(file=fixed_mjcf),
    )
    os.unlink(fixed_mjcf)  # delete the temp xml now that genesis has read it

    # tray in front of the robot at roughly where the hands would be
    # its not attached to anything yet so itll just fall, thats fine for now
    tray = scene.add_entity(
        gs.morphs.Box(
            size=(0.35, 0.25, 0.01),  # 35cm wide, 25cm deep, 1cm thick
            pos=(0.35, 0.0, 1.0),
            fixed=False,
        ),
        surface=gs.surfaces.Default(color=(0.6, 0.4, 0.2, 1.0)),  # brown
    )

    # ball sitting on top of the tray
    # z = tray z + half thickness + ball radius
    ball = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.04,
            pos=(0.35, 0.0, 1.06),
            fixed=False,  # will roll around, thats the whole point eventually
        ),
        surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2, 1.0)),  # red
    )

    # after build() you cant add anything new to the scene
    scene.build()

    print("scene built")
    print(f"robot dofs: {robot.n_dofs}")  # should be 29
    print(f"joints: {[j.name for j in robot.joints]}")

    # run forever so the window doesnt close immediately
    # ctrl+c to stop
    print("running... ctrl+c to quit")
    while True:
        scene.step()

if __name__ == "__main__":
    main()
