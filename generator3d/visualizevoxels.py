from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader
import numpy as np
import open3d as o3d
import glm
from Graphics.controls.game import Game
from Graphics.controls.camera import Camera
import pygame
import os

def adjustSize(voxels, size):
    for i in range(len(voxels)):
        voxels[i] = ((voxels[i][0] * 2 + 1) / size - 1, (voxels[i][1] * 2 + 1) / size - 1, (voxels[i][2] * 2 + 1) / size - 1)

def getVoxelLength(voxels, idx):
    for i in range(len(voxels[0])):
        if voxels[0][i] > idx:
            return i

def run(data, size):
    voxels = data.coalesce().indices().numpy()
    length = getVoxelLength(voxels, 0)
    voxels = voxels[1:, :length].transpose()

    game = Game()
    camera = Camera(Camera.PERSPECTIVE_VIEW)
    visualizer = VisualizeVoxels(size)
    voxels = voxels.astype(np.float32)
    adjustSize(voxels, size)
    
    visualizer.showVoxels(voxels)
    glEnableClientState(GL_VERTEX_ARRAY)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    return

        game.loopBeginning()
        camera.checkControls()
        (projection, view, model) = camera.getWorldView()
        visualizer.renderVoxels(camera.transform, projection, view, model, voxels)
        game.loopEnd(60)


class VisualizeVoxels:
    def __init__(self, size):
        self.meshVAO = -1

        cs = 1.0/size

        self.cubeVerts = np.array((
            # front
            (-cs, -cs,  cs), ( cs, -cs,  cs), ( cs,  cs,  cs),
            ( cs,  cs,  cs), (-cs,  cs,  cs), (-cs, -cs,  cs),
            # right
            ( cs, -cs,  cs), ( cs, -cs, -cs), ( cs,  cs, -cs),
            ( cs,  cs, -cs), ( cs,  cs,  cs), ( cs, -cs,  cs),
            # back
            (-cs,  cs, -cs), ( cs,  cs, -cs), ( cs, -cs, -cs),
            ( cs, -cs, -cs), (-cs, -cs, -cs), (-cs,  cs, -cs),
            # left
            (-cs, -cs, -cs), (-cs, -cs,  cs), (-cs,  cs,  cs),
            (-cs,  cs,  cs), (-cs,  cs, -cs), (-cs, -cs, -cs),
            # bottom
            (-cs, -cs, -cs), ( cs, -cs, -cs), ( cs, -cs,  cs),
            ( cs, -cs,  cs), (-cs, -cs,  cs), (-cs, -cs, -cs),
            # top
            (-cs,  cs,  cs), ( cs,  cs,  cs), ( cs,  cs, -cs),
            ( cs,  cs, -cs), (-cs,  cs, -cs), (-cs,  cs,  cs)
        ), dtype=np.float32)

        vertVoxShader = """
            #version 330 core
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 offset;
            layout (location = 2) in vec3 aNormal;

            uniform mat4 transform;
            uniform float offs;
            uniform mat4 projection;
            uniform mat4 view;
            uniform mat4 model;
            out vec3 Normal;
            out vec3 FragPos;
            out vec3 camera;

            void main()
            {
                gl_Position = projection * view * model * transform * vec4(position + offset, 1.0);
                Normal = vec3(transform * vec4(aNormal, 1.0));
                FragPos = vec3(projection * view * model * transform * vec4(position + offset, 1.0));
                camera = vec3(0,0,-0.3);
            }
        """

        fragVoxShader = """
            #version 330 core
            layout(location = 0) out vec4 color;
            uniform vec3 lightPos;
            in vec3 FragPos;
            in vec3 Normal;  
            in vec3 camera;

            float distance(vec3 p1, vec3 p2) {
                vec3 diff = p1 - p2;
                return sqrt(dot(diff, diff));
            }

            float absVec(vec3 p) {
                return sqrt(dot(p, p));
            }
            
            void main()
            {
                vec3 norm = normalize(Normal);
                vec3 lightDir = normalize(camera);
                float dist = min(2 / distance(camera, FragPos), 1) * 0.6;
                float col = max(dot(norm, lightDir) * 0.4 + dist, 0.2);
                color = vec4(vec3(col), 1.0);
            }
        """

        self.shaderProgram = glCreateProgram()
        vertShader = compileShader(vertVoxShader, GL_VERTEX_SHADER)
        glAttachShader(self.shaderProgram, vertShader)
        fragShader = compileShader(fragVoxShader, GL_FRAGMENT_SHADER)
        glAttachShader(self.shaderProgram, fragShader)
        glLinkProgram(self.shaderProgram)

    def showVoxels(self, voxs):
        print(len(voxs))
        cubeVs = self.cubeVerts

        self.meshVAO = glGenVertexArrays(1)
        instanceVBO = glGenBuffers(1)
        vertexVBO = glGenBuffers(1)
        normalVBO = glGenBuffers(1)


        glBindVertexArray(self.meshVAO)

        glBindBuffer(GL_ARRAY_BUFFER, vertexVBO)
        glBufferData(GL_ARRAY_BUFFER, cubeVs.nbytes, cubeVs, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (ctypes.c_void_p(0)))

        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO)
        glBufferData(GL_ARRAY_BUFFER, voxs.nbytes, voxs, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)

        norms = []
        for i in range(0, len(cubeVs), 3):
            normal = np.cross(cubeVs[i+1]-cubeVs[i], cubeVs[i+2]-cubeVs[i])
            norms.append(normal)
            norms.append(normal)
            norms.append(normal)


        normals = np.array(norms, dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, normalVBO)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), ctypes.c_void_p(0))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
    
    def renderVoxels(self, transform, projection, view, model, voxs):
        glUseProgram(self.shaderProgram)

        transform_location = glGetUniformLocation(self.shaderProgram, "transform")
        glUniformMatrix4fv(transform_location, 1, GL_TRUE, transform)

        # get uniform locations
        projection_loc = glGetUniformLocation(self.shaderProgram, "projection")
        view_loc = glGetUniformLocation(self.shaderProgram, "view")
        model_loc = glGetUniformLocation(self.shaderProgram, "model")
        light_loc = glGetUniformLocation(self.shaderProgram, "lightPos")

        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection))
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))

        camera_position = glm.inverse(transform)[3]
        #print(transform)

        # Access the camera position components
        camera_x = camera_position.x
        camera_y = camera_position.y
        camera_z = camera_position.z


        glUniform3f(light_loc, camera_x, camera_y, camera_z)


        glBindVertexArray(self.meshVAO)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 36, len(voxs))
        glBindVertexArray(0)
        glUseProgram(0)

