from OpenGL.GL import glEnableClientState, GL_VERTEX_ARRAY
import time
from Graphics.voxelize.voxalize import Voxelize
from Graphics.models.cube import TestCube
from Graphics.models.objloader import ObjLoader
from Graphics.controls.game import Game
from Graphics.controls.camera import Camera

def run ():
    # Initialize Pygame and OpenGL
    game = Game()
    camera = Camera(Camera.PERSPECTIVE_VIEW)
    vox = Voxelize(10)
    tcube = TestCube()
    #tcube.rotateCube()
    #obj = ObjLoader("..\\DataSets\\MegaScans\\OBJ\\fern82.obj")

    start = time.time()
    vox.voxelize(tcube.vertices, tcube.surfaces)
    #vox.voxelize(obj.vertices, None)
    end = time.time()
    print(f'Voxel time {time.time() - start}')

    vox.showVoxels()
    glEnableClientState(GL_VERTEX_ARRAY)

    # Render the OBJ file
    while True:
        game.loopBeginning()
        camera.checkControls()

        (projection, view, model) = camera.getWorldView()

        #obj.renderObj(camera.transform, projection, view, model)
        tcube.renderCube(camera.transform, projection, view, model)
        vox.renderVoxels(camera.transform, projection, view, model)
        
        game.loopEnd(60)



'''
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pywavefront import Wavefront

# Load OBJ file
obj = Wavefront('Graphics\\test.obj')

# Initialize Pygame and OpenGL
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, pygame.DOUBLEBUF|pygame.OPENGL)

# Setup OpenGL
gluPerspective(45, (display[0]/display[1]), 0.1, 500.0)
glTranslatef(0.0, 0.0, -150)

# Render the OBJ file
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        glRotatef(1, 0, -1, 0)
    if keys[pygame.K_RIGHT]:
        glRotatef(1, 0, 1, 0)
    if keys[pygame.K_UP]:
        glRotatef(1, -1, 0, 0)
    if keys[pygame.K_DOWN]:
        glRotatef(1, 1, 0, 0)

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)


    glBegin(GL_TRIANGLES)
    for name, material in obj.materials.items():
        glColor(material.diffuse[:3])
        for face in obj.vertices:
            glVertex(face)
    glEnd()

    pygame.display.flip()
    pygame.time.wait(10)






vertices = (
    ( 1, -1, -1),
    ( 1,  1, -1),
    (-1,  1, -1),
    (-1, -1, -1),
    ( 1, -1,  1),
    ( 1,  1,  1),
    (-1, -1,  1),
    (-1,  1,  1)
)

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
)

surfaces = (
    (0,1,2),
    (0,2,3),
    (4,0,3),
    (4,3,6),
    (5,4,6),
    (5,6,7),
    (1,5,7),
    (1,7,2),
    (4,5,1),
    (4,1,0),
    (2,7,6),
    (2,6,3)
)

colors = (
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (1,1,0),
    (1,0,1),
    (0,1,1),
    (1,1,1),
    (0,0,0)
)

def Cube():
    glBegin(GL_TRIANGLES)
    for surface in surfaces:
        x = 0
        for vertex in surface:
            x+=1
            glColor3fv(colors[x])
            glVertex3fv(vertices[vertex])
    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    glTranslatef(0.0,0.0,-5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            glRotatef(1, 0, -1, 0)
        if keys[pygame.K_RIGHT]:
            glRotatef(1, 0, 1, 0)
        if keys[pygame.K_UP]:
            glRotatef(1, -1, 0, 0)
        if keys[pygame.K_DOWN]:
            glRotatef(1, 1, 0, 0)

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube()
        pygame.display.flip()
        pygame.time.wait(10)

main()
'''