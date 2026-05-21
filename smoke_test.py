import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.0, 0.0, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    show_viewer=True,
)
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
scene.build()

for _ in range(2000):
    scene.step()
