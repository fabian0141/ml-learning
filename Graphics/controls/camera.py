import pygame
import math
import numpy as np
import glm
from Graphics.helper.mathext import getRotation4

class Camera:
    ORTHO_VIEW = 0
    PERSPECTIVE_VIEW = 1

    def __init__(self, view, viewSize = (1400, 1050)):
        self.viewType = view
        self.rotation = glm.vec3(0,0,0)
        self.translation = glm.vec3(0,0,0)
        self.aspectRatio = viewSize[0] / viewSize[1]
        self.transform = getRotation4(self.rotation)

    def checkControls(self, speed = 1):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.rotation.y = (self.rotation.y - 1) % 360
        if keys[pygame.K_RIGHT]:
            self.rotation.y = (self.rotation.y + 1) % 360
        if keys[pygame.K_UP]:
            self.rotation.x = (self.rotation.x - 1) % 360
        if keys[pygame.K_DOWN]:
            self.rotation.x = (self.rotation.x + 1) % 360
        if keys[pygame.K_w]:
            self.translation.z += 0.1 * speed
        if keys[pygame.K_s]:
            self.translation.z -= 0.1 * speed

        self.transform = getRotation4(self.rotation)


    def getWorldView(self, rotation = glm.vec3(-1, 0, 0)):
        match self.viewType:
            case self.ORTHO_VIEW:
                size = 1.1 - self.translation.z / 2
                return (glm.ortho(-size,size,-size,size,0.1,500),
                    glm.lookAt(glm.add(rotation, glm.vec3(0, 0, 0)), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0)),
                    glm.rotate(glm.mat4(1.0), glm.radians(0.0), rotation))
            case self.PERSPECTIVE_VIEW:
                return (glm.perspective(glm.radians(45.0), self.aspectRatio, 0.1, 5000.0), 
                    glm.lookAt(glm.vec3(0, 0, -2.9 + self.translation.z) , glm.vec3(0, 0, 0), glm.vec3(0, 1, 0)),
                    glm.rotate(glm.rotate(glm.mat4(1.0), glm.radians(10.0), glm.vec3(0,1,0)), glm.radians(20.0), rotation))


    