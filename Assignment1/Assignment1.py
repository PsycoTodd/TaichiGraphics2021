import taichi as ti

ti.init(arch=ti.gpu)

n = 320
PI = 3.1415926
pixels = ti.field(dtype = float, shape = (n, n, 3))

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])


@ti.kernel
def render(t: float):
    for i, j, k in pixels:
        c = ti.Vector([0.285, ti.sin(t  + k * PI / 3.0 ) * 0.01])
        z = ti.Vector([i / n - 0.5, j / n - 0.5]) * 2
        iter = 0
        while z.norm() < 20 and iter < 50:
            z = complex_sqr(z) + c
            iter += 1
        pixels[i, j, k] = iter * 0.02

def main(output_img=False):
    gui = ti.GUI("Julia Set Todd", res=(n, n))
    for ts in range(1000000):
        if gui.get_event(ti.GUI.ESCAPE):
            exit()
        
        render(ts * 0.03)
        gui.set_image(pixels)
        if output_img:
            gui.show(f'{ts:04d}.jpg')
        else:
            gui.show()

if __name__ == '__main__':
    main(output_img=False)