import taichi as ti
import numpy as np
import tina
# constants
G = 1
PI = 3.1415926

@ti.data_oriented
class CelestialObject:
    def __init__(self, N, mass, size, color) -> None:
        self.n = N
        self.m = mass
        self.dim = 3
        self.pos = ti.Vector.field(self.dim, ti.f32, shape=self.n)
        self.vel = ti.Vector.field(self.dim, ti.f32, shape=self.n)
        self.force = ti.Vector.field(self.dim, ti.f32, shape=self.n)
        self.particles = tina.SimpleParticles() # we use tina to hold the displayable particles
        self.size = size
        self.color = color

    def getRenderable(self):
        return self.particles

    @ti.func
    def Pos(self):
        return self.pos
    
    @ti.func
    def Mass(self):
        return self.m

    @ti.func
    def Number(self):
        return self.n

    @ti.func
    def clearForce(self):
        for i in self.force:
            self.force[i] = ti.Vector(np.zeros(self.dim))

    @ti.kernel
    def kernelInitialize(self, center_x: ti.f32, center_y: ti.f32, center_z: ti.f32, size: ti.f32, init_speed: ti.f32):
        for i in range(self.n):
            if self.n == 1:
                self.pos[i] = ti.Vector([center_x, center_y, center_z])
                self.vel[i] = ti.Vector(np.zeros(3))
            else:
                theta, r = self.generateThetaAndR(i, self.n)
                offset_dir = ti.Vector([ti.cos(theta), ti.sin(theta), 0])
                center = ti.Vector([center_x, center_y, center_z])
                self.pos[i] = center + r * offset_dir * size
                self.vel[i] = ti.Vector([-offset_dir[1], offset_dir[0], 0]) * init_speed

    def initialize(self, center_x, center_y, center_z, size, init_speed):
        self.kernelInitialize(center_x, center_y, center_z, size, init_speed)
        # have to initialize here
        self.particles.set_particles(np.zeros((self.n, self.dim)).astype(np.float32))
        self.particles.set_particle_radii(np.ones(self.n).astype(np.float32) * self.size)
        self.particles.set_particle_colors(np.full((self.n, len(self.color)), self.color).astype(np.float32))


    @ti.kernel
    def computeForce(self):
        self.clearForce()
        for i in range(self.n):
            p = self.pos[i]
            for j in range(self.n):
                if j != i:
                    diff = self.pos[j] - p
                    r = diff.norm(1e-2)
                    self.force[i] += 6 * self.Mass() * self.Mass() * diff / r**3
                
    def update(self, h):
        self.kernelUpdate(h)
         # update the renderable also
        self.particles.set_particles(self.pos.to_numpy())
    
    @ti.kernel
    def kernelUpdate(self, h: ti.f32):
        for i in self.vel:
            self.vel[i] += h * self.force[i] / self.Mass()
            self.pos[i] += h * self.vel[i]

@ti.data_oriented
class Star(CelestialObject):
    def __init__(self, N, mass, size, color) -> None:
        super().__init__(N, mass, size, color)
        pass

    @staticmethod
    @ti.func
    def generateThetaAndR(i, n):
        theta = 2 * PI * i / ti.cast(n, ti.f32)
        r = 1
        return theta, r

@ti.data_oriented
class Planet(CelestialObject):
    def __init__(self, N, mass, size, color) -> None:
        super().__init__(N, mass, size, color)
        pass

    @staticmethod
    @ti.func
    def generateThetaAndR(i, n):
        theta = 2 * PI * ti.random()
        r = (ti.sqrt(ti.random()) * 0.4 + 0.6)
        return theta, r

    @ti.kernel
    def computeForce(self, stars: ti.template()):
        self.clearForce()
        for i in range(self.n):
            p = self.pos[i]

            for j in range(self.n):
                if  i != j:
                    diff = self.pos[j] - p
                    r = diff.norm(1e-2)
                    self.force[i] += G * self.Mass() * self.Mass() * diff / r**3

            for j in range(stars.Number()):
                diff = stars.Pos()[j] - p
                r = diff.norm(1e-2)
                self.force[i] += G * self.Mass() * stars.Mass() * diff / r**3

    
