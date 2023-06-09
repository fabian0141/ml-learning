import os; 
os.environ["NUMBA_ENABLE_CUDASIM"] = "0"; 
import numba
import numba.cuda as cuda
from numba.experimental import jitclass
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader
import numpy as np
import glm
import math
import time
from Graphics.models.objloader import ObjLoader
from multiprocessing import Pool
import threading

threads = 512

def voxelizeObj(name):
    start = time.time()
    savePath = "..\\DataSets\\MegaScans\\Voxel128"
    obj = ObjLoader("..\\DataSets\\MegaScans\\OBJ\\" + name)

    curTime = time.time()
    objLoadTime = curTime - start
    voxelizer = CuVoxelizer(64, True)
    voxelizer.voxelize(obj.vertices, [], savePath + "\\" + name.split(".")[0], views)
    print(f'Voxelize {name} {objLoadTime} {time.time() - curTime}')

def getRotationMatrix(vec):
    # Normalize the vector
    vector = vec / np.linalg.norm(vec)

    # Define the Z-axis vector
    z_axis = np.array([0, 0, 1], dtype=np.float64)

    # Calculate the cross product and the dot product
    axis = np.cross(z_axis, vector)
    normalize = np.linalg.norm(axis)
    if (normalize != 0):
        axis = axis / normalize
    dot_product = np.dot(vector, z_axis)

    # Calculate the angle between the vectors
    angle = np.arccos(dot_product)

    # Construct the rotation matrix
    skew_symmetric = np.array([[0, -axis[2], axis[1]],
                               [axis[2], 0, -axis[0]],
                               [-axis[1], axis[0], 0]], dtype=np.float64)

    rotation_matrix = (np.cos(angle) * np.eye(3) +
                       (1 - np.cos(angle)) * np.outer(axis, axis) +
                       np.sin(angle) * skew_symmetric)
    

    #newVec = np.dot(rotation_matrix, (1,0,0))
    #length = np.linalg.norm(newVec)
    #length3 = np.linalg.norm(axis)
    #length4 = np.linalg.norm(vector)

    #testVec = np.dot(rotation_matrix, (0,0,1))
    #testVec2 = np.dot(rotation_matrix, vector)

    #if abs(length - 1.0) > 1e-12:
    #    raise Exception(vec, newVec, length)
        #raise Exception(np.sum(rotation_matrix))

    return rotation_matrix

def initPool(the_int):
    global views
    views = the_int

def run():
    start = time.time()
    path = "..\\DataSets\\MegaScans\\OBJ"
    names = os.listdir(path)

    icosphere = ObjLoader("C:\\Users\\Fabian\\Documents\\AI_Development\\DataSets\\icosphere.obj")
    vecs = np.array(icosphere.points, dtype=np.float64)
    rotationMatrices = []
    for v in vecs:
        rotationMatrices.append(getRotationMatrix(v))

    #initPool(rotationMatrices)
    #voxelizeObj(names[30])

    with Pool(16, initPool, (rotationMatrices,)) as p:
        p.map(voxelizeObj, names)

    print(f'Total Voxelization {time.time() - start}')


@cuda.jit(device=True)
def place(pos, grid):
    cuda.atomic.or_(grid, (pos[0], pos[1], pos[2] // 64), 1 << (pos[2] % 64))
    #grid[pos[0]][pos[1]][pos[2] // 64] = grid[pos[0]][pos[1]][pos[2] // 64] + (1 << (pos[2] % 64))
    
@cuda.jit(device=True)
def computeGCD(x, y, z):
    def gcd(x, y):
        if (x == 0):
            return x
        while(y and abs(y) >= 1e-12):
            x, y = y, x % y
        return abs(x)
    return gcd(gcd(x,y),z)

@cuda.jit(device=True)
def roundFast(num, gridSize):
    return int(num * gridSize + gridSize + 0.5)
    
@cuda.jit(device=True)
def getSteps(p1, p2, gridSize):
    x = abs(p2[0] - p1[0])
    y = abs(p2[1] - p1[1])
    z = abs(p2[2] - p1[2])

    gcd = computeGCD(x,y,z)
    if (gcd == 1 or gcd <= 1e-12):
        gcd = 0
    return max(1, int((x+y+z-gcd) * gridSize))

@cuda.jit(device=True)
def dotProduct(view, verts, idx):
    vec = cuda.local.array(3, dtype=numba.float64)  # Allocate a typed array on the device
    for i in range(3):
        vec[i] = 0.0

    for i in range(3):
        for j in range(3):
            vec[i] += view[i][j] * verts[idx][j]
    
    for i in range(3):
        verts[idx][i] = vec[i]
    

@cuda.jit(device=True)
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

@cuda.jit(device=True)
def adjustSize(verts, minVert, maxVert, start, end):
    center = ((maxVert[0] + minVert[0]) / 2, (maxVert[1] + minVert[1]) / 2, (maxVert[2] + minVert[2]) / 2 )
    grow = 2/max(maxVert[0] - minVert[0], maxVert[1] - minVert[1], maxVert[2] - minVert[2])

    for i in range(start, end):
        if i >= len(verts):
            break
        verts[i] = ((verts[i][0] - center[0]) * grow, (verts[i][1] - center[1]) * grow, (verts[i][2] - center[2]) * grow)

@cuda.jit(device=True)
def setGridToZero(grid, size, idx):
    gridSlice = size**2
    start = idx * gridSlice // threads
    end = min((idx+1) * gridSlice // threads, gridSlice)

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
            for k in range(len(grid[i][j])):
                grid[i][j][k] = 0
        
        sJ = 0

@cuda.jit
def voxelizeVertsParallel(grid, verts, gridSize, result, polyAmount, view, zeroGrid):
    size = gridSize * 2 + 1
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = (tx + ty * bw) * polyAmount * 3
    end = pos + polyAmount * 3
    
    if not zeroGrid:
        setGridToZero(grid, size, tx)


    minVerts = cuda.shared.array(shape=3, dtype=numba.float64)
    maxVerts = cuda.shared.array(shape=3, dtype=numba.float64)

    minVert, maxVert = transformModel(verts, view, pos, end)
    for i in range(3):
        cuda.atomic.min(minVerts, i, minVert[i])
        cuda.atomic.max(maxVerts, i, maxVert[i])

    cuda.syncthreads()
    adjustSize(verts, minVerts, maxVerts, pos, end)
    
    for _ in range(polyAmount):
        pos += 3
        if pos + 3 >= len(verts):
            break
        
        voxelizeTriangle(verts[pos], verts[pos+1], verts[pos+2], gridSize, grid)

    cuda.syncthreads()

    reduceGrid(size, gridSize, grid, tx, result)

@cuda.jit
def voxelizeTrisParallel(tris, verts, gridSize, grid):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw

    if pos >= len(tris):
        return

    tri = tris[pos]
    v1  = verts[tri[0]]
    v2 = verts[tri[1]]
    v3 = verts[tri[2]]

    voxelizeTriangle(v1,v2,v3,gridSize, grid)

@cuda.jit(device=True)
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

@cuda.jit(device=True)
def comp(vec1, vec2):
    return vec1[0] == vec2[0] and vec1[1] == vec2[1] and vec1[2] == vec2[2]

@cuda.jit(device=True)
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
    totalStepsY = allSteps[1][0]  * 3# multiply by two to reduce artifacts on very smooth faces
    #totalStepsY = allSteps[1][0] * 2

    deltaX = (x2[0]-x1[0], x2[1]-x1[1], x2[2]-x1[2])
    deltaY = (x3[0]-x1[0], x3[1]-x1[1], x3[2]-x1[2])


    for stepY in range(totalStepsY):
        scal = stepY/totalStepsY
        
        startPoint = cuda.local.array(shape=3, dtype=np.float64)
        endPoint = cuda.local.array(shape=3, dtype=np.float64)

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

@cuda.jit(device=True)
def reduceGrid(size, gridSize, grid, idx, result):
    size -= 1
    sizes = cuda.shared.array(shape=threads, dtype=numba.int32)
    sizesAcuumulated = cuda.shared.array(shape=16, dtype=numba.int32)

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
                if grid[i][j][k // 64] & numba.uint64(1 << (k % 64)):
                    counter += 1
        
        sJ = 0

    sizes[idx] = counter
    for i in range(idx // 32 + 1, 16):
        cuda.atomic.add(sizesAcuumulated, i, counter)

    cuda.syncthreads()
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
                if grid[i][j][k // 64] & numba.uint64(1 << (k % 64)):
                    #result[resultPos] = ((i * 2 + 1) / size - 1, (j * 2 + 1) / size - 1, (k * 2 + 1) / size - 1)
                    result[resultPos] = (i, j, k)
                    resultPos += 1
            
            for k in range(len(grid[i][j])):
                grid[i][j][k] = 0

        sJ = 0

    if idx == 511:
        result[0] = (startResult + sizes[511], 0, 0)

@cuda.jit
def convToList(gridSize, smgrid, arr):
    counter = 1
    for i in range(len(smgrid)):
        for j in range(1, smgrid[i][0][0]):
            val = smgrid[i][j]
            arr[counter] = ((val[0] * 2 + 1) / gridSize - 1, (val[1] * 2 + 1) / gridSize - 1, (val[2] * 2 + 1) / gridSize - 1)
            counter += 1

    arr[0] = (counter, 0, 0)


class CuVoxelizer:

    def __init__(self, gridSize, onlyVoxelize = False):
        self.gridSize = gridSize
        #self.voxels = {}

        self.roundDevider = int(math.log10(gridSize))

        size = gridSize * 2 + 1
        #self.container = np.zeros((size, size, size))
        #self.container = np.array(self.container)

        if not onlyVoxelize:
            self.meshVAO = -1
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

    def voxelize(self, ogVerts, tris, path="..\\DataSets\\MegaScans\\voxel", views = [(1,0,0), (0,1,0), (0,0,1)]):
        threadsperblock = threads
        #start = time.time()
        size = self.gridSize * 2 + 1
        grid = cuda.device_array(shape=(size, size, (size - 1) // 64 + 1), dtype=np.uint64)
        #grid = np.zeros((size, size, (size - 1) // 64 + 1), dtype=np.uint64)

        index = 0
        for view in views:
            #grid = np.zeros((size, size, (size - 1) // 64 + 1), dtype=np.uint64)

            result = np.zeros((1000000, 3), dtype=np.uint32)
            verts = np.copy(ogVerts)

            if len(tris) == 0:
                blockspergrid = (verts.shape[0] + (threadsperblock - 1)) // threadsperblock
                voxelizeVertsParallel[1, threadsperblock](grid, verts, self.gridSize, result, blockspergrid, view, index > 0)
            else:
                blockspergrid = (tris.size + (threadsperblock - 1)) // threadsperblock
                voxelizeTrisParallel[blockspergrid, threadsperblock](tris, verts, self.gridSize, grid)

            #spacer = '\\'
            #print(f'Voxelize {path.split(spacer)[-1]} {time.time() - start}')
            result = result[1:int(result[0][0])]

            self.voxs = np.array(result, dtype=np.uint8)
            np.save(f'{path}_{index}', self.voxs)
            index += 1




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




#def run():
#    testArr = np.array([1] * 32000000)
#    threadsperblock = 32
#    blockspergrid = (testArr.size + (threadsperblock - 1)) // threadsperblock
#    increment_by_one[blockspergrid, threadsperblock](testArr)
#    print(testArr)


#@cuda.jit
#def increment_by_one(testArr):
#    # Thread id in a 1D block
#    tx = cuda.threadIdx.x
#    # Block id in a 1D grid
#    ty = cuda.blockIdx.x
#    # Block width, i.e. number of threads per block
#    bw = cuda.blockDim.x
#    # Compute flattened index inside the array
#    pos = tx + ty * bw
#    if pos < testArr.size:  # Check array boundaries
#        testArr[pos] += 1