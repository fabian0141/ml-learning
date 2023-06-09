import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader
import glm
import math

threads = 512

def place(pos, grid):
    grid[pos[0]][pos[1]][pos[2] // 64] = grid[pos[0]][pos[1]][pos[2] // 64] + (1 << (pos[2] % 64))

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


def dotProduct(view, verts, idx):
    vec = (0.0, 0.0, 0.0)

    for i in range(3):
        for j in range(3):
            vec[i] += view[i][j] * verts[idx][j]
    
    for i in range(3):
        verts[idx][i] = vec[i]
    

def transformModel(verts, view, start, end):
    minVert = (10.0, 10.0, 10.0)
    maxVert = (-10.0, -10.0, -10.0)

    for i in range(start, end):
        if i >= len(verts):
            break
        
        dotProduct(view, verts, i)
        minVert = (min(verts[i][0], minVert[0]), min(verts[i][1], minVert[1]), min(verts[i][2], minVert[2]))
        maxVert = (max(verts[i][0], maxVert[0]), max(verts[i][1], maxVert[1]), max(verts[i][2], maxVert[2]))

    return minVert, maxVert

def adjustSize(verts, minVert, maxVert, start, end):
    center = ((maxVert[0] + minVert[0]) / 2, (maxVert[1] + minVert[1]) / 2, (maxVert[2] + minVert[2]) / 2 )
    grow = 2/max(maxVert[0] - minVert[0], maxVert[1] - minVert[1], maxVert[2] - minVert[2])

    for i in range(start, end):
        if i >= len(verts):
            break
        verts[i] = ((verts[i][0] - center[0]) * grow, (verts[i][1] - center[1]) * grow, (verts[i][2] - center[2]) * grow)

def voxelizeVertsParallel(grid, verts, gridSize, result, polyAmount, i, sizes, sizesAcuumulated):
    size = gridSize * 2 + 1
    tx = i
    pos = tx * polyAmount * 3
    end = pos + polyAmount * 3

    minVerts = [0.0] * 3
    maxVerts = [0.0] * 3

    #minVert, maxVert = transformModel(verts, view, pos, end)
    #for i in range(3):
    #    minVerts[i] = min(minVerts[i], minVert[i])
    #    maxVerts[i] = max(maxVerts[i], maxVert[i])

    #adjustSize(verts, minVerts, maxVerts, pos, end)

    for _ in range(polyAmount):
        pos += 3
        if pos + 3 >= len(verts):
            break
        
        voxelizeTriangle(verts[pos], verts[pos+1], verts[pos+2], gridSize, grid)



    reduceGrid(size, gridSize, grid, tx, result, sizes, sizesAcuumulated)

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
    totalStepsY = allSteps[1][0] * 3# multiply by two to reduce artifacts on very smooth faces
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

def reduceGrid(size, gridSize, grid, idx, result, sizes, sizesAcuumulated):
    size -= 1

    gridSlice = size**2
    start = idx * gridSlice // threads
    end = min((idx+1) * gridSlice // threads, gridSlice)
    counter = 0
    counter2 = 0


    startI = start // size
    endI = (end + size - 1) // size
    startJ = start % size
    endJ = (end + size - 1) % size + 1
    sJ = startJ
    eJ = size

    for i in range(startI, endI):
        if (i == endI - 1):
            eJ = endJ

        for j in range(sJ, eJ):
            for k in range(size):
                counter2 += 1
                if grid[i][j][k // 64] & np.uint64(1 << (k % 64)):
                    counter += 1
        
        sJ = 0

    sizes[idx] = counter
    for i in range(idx // 32 + 1, 16):
        sizesAcuumulated[i] += counter

    #result[idx+1] = (float(counter), float(counter2), float(idx))
    #result[idx+1] = (float(startI), float(endI), float(startJ))
    #result[idx+1] = (float(sizesAcuumulated[idx // 32]), 0.0, 0.0)

    startResult = 1 + sizesAcuumulated[idx // 32]
    for i in range(idx // 32 * 32, idx):
        startResult += sizes[i]

    #result[idx+1] = (float(startResult), float(sizes[idx]), float(sizesAcuumulated[idx // 32]))


    resultPos = startResult
    sJ = startJ
    eJ = size

    for i in range(startI, endI):
        if (i == endI - 1):
            eJ = endJ

        for j in range(sJ, eJ):
            for k in range(size):
                if grid[i][j][k // 64] & np.uint64(1 << (k % 64)):
                    result[resultPos] = ((i * 2 + 1) / size - 1, (j * 2 + 1) / size - 1, (k * 2 + 1) / size - 1)
                    #result[resultPos] = (i, j, k)
                    resultPos += 1

        sJ = 0

    if idx == 511:
        result[0] = (startResult + sizes[511], 0, 0)

class TestVoxelizer:

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
        threadsperblock = threads
        size = self.gridSize * 2 + 1
        grid = np.zeros((size, size, (size - 1) // 64 + 1), dtype=np.uint64)
        result = np.zeros((1000000, 3), dtype=np.float32)
        sizes = [0] * threads
        sizesAcuumulated = [0] * 16

        blockspergrid = (verts.shape[0] + (threadsperblock - 1)) // threadsperblock
        for i in range(threadsperblock):
                voxelizeVertsParallel(grid, verts, self.gridSize, result, blockspergrid, i, sizes, sizesAcuumulated)

        count = 0
        for gss in grid:
            for gs in gss:
                for g in gs:
                    i = np.uint64(1)
                    for _ in range(64):
                        if np.bitwise_and(i, g):
                            count += 1
                        i = np.left_shift(i, np.uint64(1))
        print(count)

        grid2 = np.load("vox.npy")

        for i in range(len(result)):
            if (result[i] == 0).all():
                break

        result = result[1:int(result[0][0])]
        self.voxs = np.array(result, dtype=np.float32)


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