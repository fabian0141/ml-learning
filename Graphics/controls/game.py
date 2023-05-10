import pygame
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective

class Game:

    def __init__(self, width=1400, height=1050):
        pygame.init()
        pygame.display.set_mode((width, height), pygame.DOUBLEBUF|pygame.OPENGL, 24)
        self.clock = pygame.time.Clock()

        # Setup OpenGL
        gluPerspective(45, (width/height), 0.1, 500.0)
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        #glRotatef(90, -1, -1, 0)


    def loopBeginning(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        #glPointSize(200/gridSize)

    def loopEnd(self, ticks=-1):
        pygame.display.flip()
        if ticks > -1:
            self.clock.tick(ticks)