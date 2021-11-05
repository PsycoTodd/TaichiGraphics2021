import taichi as ti
import tina
from celestial_objects import Star, Planet

if __name__ == "__main__":
    ti.init(arch=ti.gpu)

    # control
    paused = False
    export_images = False

    # starts and planets
    stars = Star(N=2, mass=1000, size=0.1, color=[0.8, 0.8, 0.0])
    planets = Planet(N=100, mass=1, size=0.02, color=[1.0, 1.0, 1.0])
    # add to scene
    scene = tina.Scene(800)
    material = tina.Classic()
    scene.add_object(stars.getRenderable(), material)
    scene.add_object(planets.getRenderable(), material)
    
    stars.initialize(0.5, 0.5, 0.0, 0.2, 10)
    planets.initialize(0.5, 0.5, 0.0, 0.4, 10)

    # GUI
    my_gui = ti.GUI("Galaxy3D", (800, 800))
    h = 5e-5 # time-step size
    i = 0
    
    while my_gui.running:
        for e in my_gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                exit()
            elif e.key == ti.GUI.SPACE:
                paused = not paused
                print("paused = ", paused)
            elif e.key == 'r':
                stars.initialize(0.5, 0.5, 0.0, 0.2, 10)
                planets.initialize(0.5, 0.5, 0.4, 10)
                i = 0
            elif e.key == 'i':
                export_images = not export_images

        if not paused:
            stars.computeForce()
            planets.computeForce(stars)
            for celestial_obj in (stars, planets):
                celestial_obj.update(h)
            i += 1
        
        scene.input(my_gui)
        scene.render()
        my_gui.set_image(scene.img)
        if export_images:
            my_gui.show(f"images\output_{i:05}.png")
        else:
            my_gui.show()