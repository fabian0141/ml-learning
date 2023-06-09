import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader
import glm
import math

def place(pos, grid):
    grid[pos[0]][pos[1]][pos[2]] = True

def computeGCD(x, y, z):
    def gcd(x, y):
        if (x == 0):
            return x
        while(y and abs(y) >= 1e-12):
            x, y = y, x % y
        return abs(x)
    return gcd(gcd(x,y),z)

def roundFast(num, gridSize):
    return int(num * gridSize + gridSize + 0.5) 

def getSteps(p1, p2, gridSize):
    x = abs(p2[0] - p1[0])
    y = abs(p2[1] - p1[1])
    z = abs(p2[2] - p1[2])

    #if x + y + z < 1/gridSize:
    #    return 0

    gcd = computeGCD(x,y,z)
    if (gcd == 1 or gcd <= 1e-12):
        gcd = 0
    return max(1, int((x+y+z-gcd) * gridSize))

def voxelizeVertsParallel(verts, gridSize, grid, i, j):
    pos = (j + i * 32) * 3 # change pos

    if pos + 3 >= len(verts):
        return
    voxelizeTriangle(verts[pos], verts[pos+1], verts[pos+2], gridSize, grid)

def voxelizeTrisParallel(tris, verts, gridSize, grid, i, j):
    pos = j + i * 32 # change pos
    if pos >= len(tris):
        return

    tri = tris[pos]
    v1  = verts[tri[0]]
    v2 = verts[tri[1]]
    v3 = verts[tri[2]]

    voxelizeTriangle(v1,v2,v3,gridSize, grid) 

def sorted(arr):
    if arr[0][0] <= arr[1][0]:
        if arr[1][0] <= arr[2][0]:
            return (arr[0], arr[1], arr[2])
        elif arr[0][0] <= arr[2][0]:
            return (arr[0], arr[2], arr[1])
        else:
            return (arr[2], arr[0], arr[1])
    else:
        if arr[0][0] <= arr[2][0]:
            return (arr[1], arr[0], arr[2])
        elif arr[1][0] <= arr[2][0]:
            return (arr[1], arr[2], arr[0])
        else:
            return (arr[2], arr[1], arr[0])

def comp(vec1, vec2):
    return vec1[0] == vec2[0] and vec1[1] == vec2[1] and vec1[2] == vec2[2]

def voxelizeTriangle(v1, v2, v3, gridSize, grid):

    allSteps = ((getSteps(v1,v2, gridSize), 0), (getSteps(v1,v3, gridSize), 1), (getSteps(v2,v3, gridSize), 2))
    allSteps = sorted(allSteps)
    allVecs = ((v1, v2, v3), (v1, v3, v2), (v2, v3, v1))

    #get starting position
    if comp(allVecs[allSteps[0][1]][1], allVecs[allSteps[1][1]][1]) or comp(allVecs[allSteps[0][1]][1], allVecs[allSteps[1][1]][2]):
        x1 = allVecs[allSteps[0][1]][0]
        x2 = allVecs[allSteps[0][1]][1]
        x3 = allVecs[allSteps[0][1]][2]
    else:
        x1 = allVecs[allSteps[0][1]][1]
        x2 = allVecs[allSteps[0][1]][0]
        x3 = allVecs[allSteps[0][1]][2]

    totalStepsX = allSteps[0][0]
    totalStepsY = allSteps[1][0] # multiply by two to reduce artifacts on very smooth faces
    #totalStepsY = allSteps[1][0] * 2

    deltaX = (x2[0]-x1[0], x2[1]-x1[1], x2[2]-x1[2])
    deltaY = (x3[0]-x1[0], x3[1]-x1[1], x3[2]-x1[2])


    for stepY in range(totalStepsY):
        scal = stepY/totalStepsY
        
        startPoint = np.zeros(3, dtype=np.float64)
        endPoint = np.zeros(3, dtype=np.float64)

        for i in range(3):
            startPoint[i] = x1[i] + (x3[i]-x1[i]) * scal
            endPoint[i] = x2[i] + (x3[i]-x2[i]) * scal
        totalSteps = getSteps(startPoint, endPoint, gridSize)

        for stepX in range(totalSteps+1):
            relativeStepX = stepX/totalStepsX
            relativeStepY = stepY/totalStepsY

            a = x1[0] + deltaX[0]*relativeStepX + deltaY[0]*relativeStepY
            b = x1[1] + deltaX[1]*relativeStepX + deltaY[1]*relativeStepY
            c = x1[2] + deltaX[2]*relativeStepX + deltaY[2]*relativeStepY

            x = roundFast(a, gridSize)
            y = roundFast(b, gridSize)
            z = roundFast(c, gridSize)

            place((x,y,z), grid)

def reduceGrid(gridSize, grid, smgrid, pos):
    start = (gridSize // 32) * pos
    end = (gridSize // 32) * (pos+1)
    counter = 1

    for i in range(start, end):
        for j in range(gridSize):
            for k in range(gridSize):
                if grid[i][j][k]:
                    smgrid[pos][counter] = (i, j, k)
                    counter += 1

    smgrid[pos][0] = (counter, 0, 0)

def convToList(gridSize, smgrid, arr):
    counter = 1
    for i in range(len(smgrid)):
        for j in range(1, smgrid[i][0][0]):
            val = smgrid[i][j]
            arr[counter] = ((val[0] * 2 + 1) / gridSize - 1, (val[1] * 2 + 1) / gridSize - 1, (val[2] * 2 + 1) / gridSize - 1)
            counter += 1

    arr[0] = (counter, 0, 0)

class TestVoxelizer2:

    def __init__(self, gridSize):
        self.gridSize = gridSize
        self.meshVAO = -1
        self.roundDevider = int(math.log10(gridSize))

        cs = 1.0/(gridSize*2)
        print(cs)

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
    
    def voxelize(self, verts, tris):
        threadsperblock = 32

        size = self.gridSize * 2 + 1
        #grid = cuda.device_array(shape=(size, size, (size - 1) // 8 + 1), dtype=np.byte, )
        grid = np.zeros((size, size, size), dtype=np.bool8)

        if len(tris) == 0:
            blockspergrid = (verts.shape[0] + (threadsperblock - 1)) // threadsperblock
            for i in range(blockspergrid):
                for j in range(threadsperblock):
                    voxelizeVertsParallel(verts, self.gridSize, grid, i, j)

            np.save("vox", grid)

        else:
            blockspergrid = (tris.size + (threadsperblock - 1)) // threadsperblock
            for i in range(blockspergrid):
                for j in range(threadsperblock):
                    voxelizeTrisParallel(tris, verts, self.gridSize, grid, i, j)

            np.save("vox", grid)


        #smgrid = cuda.device_array(shape=(threadsperblock, 100000, 3), dtype=np.int32)
        smgrid = np.zeros((threadsperblock, 100000, 3), dtype=np.int32)
        for i in range(threadsperblock):
            reduceGrid(size, grid, smgrid, i)

        arr = np.zeros((1000000, 3), dtype=np.float32)
        convToList(size, smgrid, arr)
        arr = arr[1:int(arr[0][0])]

        self.voxs = np.array(arr, dtype=np.float32)
        #self.voxs = np.array(list(self.container.values()), dtype=np.float32)


    def showVoxels(self):
        print(len(self.voxs))
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
        glBufferData(GL_ARRAY_BUFFER, self.voxs.nbytes, self.voxs, GL_STATIC_DRAW)
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
    
    def renderVoxels(self, transform, projection, view, model):
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
        glDrawArraysInstanced(GL_TRIANGLES, 0, 36, len(self.voxs))
        glBindVertexArray(0)
        glUseProgram(0)