from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader
import numpy as np
import glm
from Graphics.controls.game import Game
from Graphics.controls.camera import Camera
import pygame
import os
from Graphics.models.objloader import ObjLoader

def getRotationMatrix(vec):
    vector = vec / np.linalg.norm(vec)
    z_axis = np.array([0, 0, 1], dtype=np.float64)

    axis = np.cross(vector, z_axis)
    normalize = np.linalg.norm(axis)
    if (normalize != 0):
        axis = axis / normalize

    dot_product = np.dot(vector, z_axis)
    angle = np.arccos(dot_product)

    skew_symmetric = np.array([[0, -axis[2], axis[1]],
                               [axis[2], 0, -axis[0]],
                               [-axis[1], axis[0], 0]], dtype=np.float64)

    rotation_matrix = (np.cos(angle) * np.eye(3) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * skew_symmetric)

    #rotation_matrix = (np.eye(3) + skew_symmetric + np.dot(skew_symmetric, skew_symmetric) * (1 - dot_product**2) / np.dot(axis, axis))

    return rotation_matrix


def getRotationMatrix2(vec):
    vector = vec / np.linalg.norm(vec)
    z_axis = np.array([0, 0, 1], dtype=np.float64)

    axis = np.cross(vector, z_axis)
    normalize = np.linalg.norm(axis)
    if (normalize != 0):
        axis = axis / normalize

    # Calculate the rotation angle
    angle = np.arccos(np.dot(vector, z_axis))

    # Construct the rotation matrix
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    rotation_matrix = np.array([[t*axis[0]*axis[0]+c, t*axis[0]*axis[1]-s*axis[2], t*axis[0]*axis[2]+s*axis[1]],
                                [t*axis[0]*axis[1]+s*axis[2], t*axis[1]*axis[1]+c, t*axis[1]*axis[2]-s*axis[0]],
                                [t*axis[0]*axis[2]-s*axis[1], t*axis[1]*axis[2]+s*axis[0], t*axis[2]*axis[2]+c]])

    return rotation_matrix



def run():
    icosphere = ObjLoader("C:\\Users\\Fabian\\Documents\\AI_Development\\DataSets\\icosphere.obj")
    vecs = np.array(icosphere.points, dtype=np.float64)

    #vec = (0.0, 1.0, 0.0)
    vectors = ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))

    points = []
    vec = (0.0, .5, 1.0)
    rot = getRotationMatrix(vec)


    #for v in vecs:
    #    rot = getRotationMatrix(v)
    #    points.append(np.dot(rot, vec))
    for v in vecs:
        points.append(np.dot(rot, v))


    points = np.array(points, dtype=np.float32)
    print(points)
    print(rot)
    print(vec)



    game = Game()
    camera = Camera(Camera.PERSPECTIVE_VIEW)
    visualizer = TestViews()

    
    visualizer.showVoxels(points)
    glEnableClientState(GL_VERTEX_ARRAY)

    while True:
        game.loopBeginning()
        camera.checkControls()
        (projection, view, model) = camera.getWorldView()
        visualizer.renderVoxels(camera.transform, projection, view, model, points)
        game.loopEnd(60)


class TestViews:
    def __init__(self):
        self.meshVAO = -1

        cs = 1.0/32

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

            void main()
            {
                gl_Position = projection * view * model * transform * vec4(position + offset, 1.0);
                Normal = aNormal;
                FragPos = vec3(model * vec4(position, 1.0));
            }
        """

        fragVoxShader = """
            #version 330 core
            layout(location = 0) out vec4 color;
            uniform vec3 lightPos;
            in vec3 FragPos;
            in vec3 Normal;  

            void main()
            {
                vec3 norm = normalize(Normal);
                vec3 lightDir = normalize(lightPos - FragPos);
                float col = max(dot(norm, lightDir), 0.3);
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
        glUniform3f(light_loc, 1, 2, -1)


        glBindVertexArray(self.meshVAO)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 36, len(voxs))
        glBindVertexArray(0)
        glUseProgram(0)

