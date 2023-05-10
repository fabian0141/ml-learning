from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader
import numpy as np
import glm
import random

class CellularCaveGenerator:
    def __init__(self, size):
        self.size = size
        self.voxels = {} #[[[] * size[0] for _ in range(size[1])] for _ in range(size[0])]
        self.chanceToStartAlive = 0.1
        self.meshVAO = -1

        cs = 1
        self.deathLimit = 2
        self.birthLimit = 6
        self.steps = 6

        self.cubeVerts = (
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
        )

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
            out vec4 fragColor;

            void main()
            {
                gl_Position = projection * view * model * transform * vec4(position + offset, 1.0);
                Normal = aNormal;
                FragPos = vec3(model * vec4(position, 1.0));
                fragColor =  vec4((aNormal + vec3(1)) / 2, 1.0);
            }
        """

        fragVoxShader = """
            #version 330 core
            layout(location = 0) out vec4 color;
            uniform vec3 lightPos;
            in vec3 FragPos;
            in vec3 Normal;  
            in vec4 fragColor;

            void main()
            {
                vec3 norm = normalize(Normal);
                vec3 lightDir = normalize(lightPos - FragPos);
                float col = max(dot(norm, lightDir), 0.3);
                color = fragColor;
            }
        """

        self.shaderProgram = glCreateProgram()
        vertShader = compileShader(vertVoxShader, GL_VERTEX_SHADER)
        glAttachShader(self.shaderProgram, vertShader)
        fragShader = compileShader(fragVoxShader, GL_FRAGMENT_SHADER)
        glAttachShader(self.shaderProgram, fragShader)
        glLinkProgram(self.shaderProgram)

    def countNeighbours(self, x, y, z, pos):
        count = 0
        for i in range(-2, 4, 2):
            posX = pos + i * 10000000
            for j in range(-2, 4, 2):
                posXY = posX + j * 1000
                for k in range(-2, 4, 2):
                    if i == 0 and j == 0 and k == 0:
                        continue

                    posXYZ = posXY + k
                    nbX = x + i
                    nbY = y + j
                    nbZ = z + k

                    if nbX < -self.size[0] or nbY < -self.size[1] or nbZ < -self.size[2] or nbX >= self.size[0] or nbY >= self.size[1] or nbZ >= self.size[2]:
                        count += 1
                    elif posXYZ in self.voxels:
                        count += 1
        return count

    def simulateStep(self):
        new = {}
        for x in range(-self.size[0], self.size[0], 2):
            xOff = (x+self.size[0]) * 10000000
            for y in range(-self.size[1], self.size[1], 2):
                yOff = (y+self.size[1]) * 1000
                for z in range(-self.size[2], self.size[2], 2):
                    mapPos = xOff+yOff+z+self.size[2]
                    neighbourCount = self.countNeighbours(x, y, z, mapPos)

                    if mapPos in self.voxels:
                        if neighbourCount >= self.deathLimit:
                            new[mapPos] = (x, y, z)
                    else:
                        if neighbourCount > self.birthLimit:
                            new[mapPos] = (x, y, z)

        self.voxels = new

    def generateCave(self):
        for x in range(-self.size[0], self.size[0], 2):
            xOff = (x+self.size[0]) * 10000000
            for y in range(-self.size[1], self.size[1], 2):
                yOff = (y+self.size[1]) * 1000
                for z in range(-self.size[2], self.size[2], 2):
                    if random.random() < self.chanceToStartAlive:
                        self.voxels[xOff+yOff+z+self.size[2]] = (x, y, z)

        for i in range(self.steps):
            print(f"Step {i}/{self.steps}")
            self.simulateStep()

    def showCave(self):
        voxs = np.array(list(self.voxels.values()), dtype=np.float32)
        cubeVs = np.array(self.cubeVerts, dtype=np.float32)

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
    
    def renderCave(self, transform, projection, view, model):
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
        glUniform3f(light_loc, 1, 2, -1)


        glBindVertexArray(self.meshVAO)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 36, len(self.voxels))
        glBindVertexArray(0)
        glUseProgram(0)