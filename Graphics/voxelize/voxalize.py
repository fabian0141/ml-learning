from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader
import numpy as np
from Graphics.helper.mathext import computeGCD
from Graphics.hashmap.hashmap import HashMap
import glm
import time
import math

class Voxelize:
    def __init__(self, gridSize):
        self.gridSize = gridSize
        self.voxels = {}
        self.meshVAO = -1
        self.roundDevider = int(math.log10(gridSize))
        self.map = HashMap(gridSize * 2)

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

    #3d manhatten metrik
    #get all voxels which touch the polygon
    def getSteps(self, p1, p2):
        x = abs(p2[0] - p1[0])
        y = abs(p2[1] - p1[1])
        z = abs(p2[2] - p1[2])

        #TODO: improve here
        gcd = computeGCD(x,y,z)
        if (gcd == 1 or gcd <= 1e-12):
            gcd = 0
        return max(1, int((x+y+z-gcd) * self.gridSize))
        #return int((x+y+z) * self.gridSize)

    def getAllSteps(self, p1, p2):
        return int((abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) + abs(p2[2] - p1[2])) * self.gridSize) + 1

    def voxalizeEdge(self, p1, p2, totalSteps):
        edge = []
        for stepX in range(totalSteps+1):
            a = p1[0] + (p2[0]-p1[0])/totalSteps*stepX
            b = p1[1] + (p2[1]-p1[1])/totalSteps*stepX
            c = p1[2] + (p2[2]-p1[2])/totalSteps*stepX

            x = self.roundFast(a)
            y = self.roundFast(b)
            z = self.roundFast(c)
            #self.voxels[x*1000000+y*1000+z] = (x,y,z)
            vec = (x,y,z)
            #self.map.place((x,y,z))
            edge.append((x, y, z))
        return edge

    def roundFast(self, num):
        #myround = int(num * self.gridSize + self.gridSize + 0.5) - 1
        return int(num * self.gridSize + self.gridSize + 0.5)
        #return int((num+2) * self.gridSize + 0.5) / self.gridSize - 2


    def voxelize(self, verts, tris):
        #totalDictTime = 0
        #totalStepTime = 0
        #totalRoundTime = 0
        #totalCalcTime = 0
        start = time.time()
        #voxels = [[[False]*100 for i in range(100)] for j in range(100)]

        if len(tris) == 0:
            for i in range(0, len(verts), 3):
                self.voxelizeTriangle(verts[i],verts[i+1],verts[i+2])
        else:
            for t in range(len(tris)):
            #for t in range(9, 10):
                tri = tris[t]
                v1  = verts[tri[0]]
                v2 = verts[tri[1]]
                v3 = verts[tri[2]]
                #print(x1,x2,x3)
                #self.voxalizeEdge(v1,v2, self.getSteps(v1, v2))
                #self.voxalizeEdge(v1,v3, self.getSteps(v1, v3))
                #self.voxalizeEdge(v2,v3, self.getSteps(v2, v3))

                self.voxelizeTriangle(v1,v2,v3)
                #self.voxelizeTriangle4(v1,v2,v3)

        end = time.time()
        #print(f'Voxel time: {end - start} totalStepTime: {totalStepTime} totalCalcTime: {totalCalcTime} totalRoundTime: {totalRoundTime} totalDictTime: {totalDictTime}')
        print(f'Voxel time: {end - start}')
        #print(len(self.voxels))

    def getDir(self, v1, v2):
        normal = (abs(v1[1]*v2[2] - v1[2]*v2[1]), abs(v1[2]*v2[0] - v1[0]*v2[2]), abs(v1[0]*v2[1] - v1[1]*v2[0]))
        if normal[0] < normal[1]:
            if normal[1] < normal[2]:
                return (2, 0, 1)
            else:
                return (1, 0, 2)
        else:
            if normal[0] < normal[2]:
                return (2, 0, 1)
            else:
                return (0, 1, 2)

    def getDelta(self, v1, v2):
        return (v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2])


    def getCorner(self, vec1, vec2, start, end, mode, dir):
        if mode == 0:
            #a = vec1[1] / vec2[1]
            #b = (vec1[0] - vec2[0] * a, 0, vec1[2] - vec2[2] * a)
            #c = vec1[2] / b[2]
            #d = (start[0] + c * b[0], start[1], end[2])

            a = vec1[dir[1]] / vec2[dir[1]]
            b = [None] * 3
            b[dir[0]], b[dir[1]], b[dir[2]] = vec1[dir[0]] - vec2[dir[0]] * a, 0, vec1[dir[2]] - vec2[dir[2]] * a

            c = vec1[dir[2]] / b[dir[2]]
            d = [None] * 3
            d[dir[0]], d[dir[1]], d[dir[2]] = start[dir[0]] + c * b[dir[0]], start[dir[1]], end[dir[2]]

        elif mode == 1:
            a = vec1[dir[2]] / vec2[dir[2]]
            b = [None] * 3
            b[dir[0]], b[dir[1]], b[dir[2]] = vec1[dir[0]] - vec2[dir[0]] * a, vec1[dir[1]] - vec2[dir[1]] * a, 0

            c = vec1[dir[1]] / b[dir[1]]
            d = [None] * 3
            d[dir[0]], d[dir[1]], d[dir[2]] = start[dir[0]] + c * b[dir[0]], end[dir[1]], start[dir[2]]

        #elif dir == 2:
        #    a = vec1[0] / vec2[0]
        #    b = (0, vec1[1] - vec2[1] * a, vec1[2] - vec2[2] * a)
        #    c = vec1[1] / b[1]
        #    d = (start[0], end[1], start[2] + c * b[2])
        
        x = self.roundFast(d[0])
        y = self.roundFast(d[1])
        z = self.roundFast(d[2])

        #self.voxels[x*1000000+y*1000+z] = (x,y,z)
        self.map.place((x,y,z))
        return (x,y,z)
    
    def getStartCorner(self, v1, v2, v3):
        chosen = [v1, v2, v3]
        for i in range(3):
            if v1[i] < v2[i]:
                if v2[i] < v3[i]:
                    chosen[1] = []
                elif v1[i] < v3[i]:
                    chosen[2] = []
                else:
                    chosen[0] = []
            else:
                if v1[i] < v3[i]:
                    chosen[0] = []
                elif v2[i] < v3[i]:
                    chosen[2] = []
                else:
                    chosen[1] = []
        sorted = [v1, v2, v3]
        for i in range(1,3):
            if len(chosen[i]) > 0:
                sorted[0], sorted[i] = sorted[i], sorted[0]
                break 
        return sorted

        

    #Find corner point
    #get normal and check which ways to calculate vectors
    #calculate vectors from corner to bounding box

    def voxelizeTriangle2(self, v1, v2, v3):

        #self.voxalizeEdge(v1, v2,  self.getSteps(v1, v2))
        #self.voxalizeEdge(v1, v3,  self.getSteps(v1, v3))
        #self.voxalizeEdge(v2, v3,  self.getSteps(v2, v3))

        vecs = self.getStartCorner(v1, v2, v3)


        d1 = self.getDelta(vecs[0], vecs[1])
        d2 = self.getDelta(vecs[0], vecs[2])

        dir = self.getDir(d1, d2)

        v = self.getCorner(d1, d2, vecs[0], vecs[1], 0, dir)
        self.voxalizeEdge(vecs[0], v, self.getSteps(vecs[0], v))

        vv = self.getCorner(d2, d1, vecs[0], vecs[2], 1, dir)
        self.voxalizeEdge(vecs[0], vv, self.getSteps(vecs[1], vv))

    def getBoundingBox(self, v1, v2, v3):
        minP = [None] * 3
        maxP = [None] * 3
        for i in range(3):
            minP[i] = min(v1[i], v2[i], v3[i])
            maxP[i] = max(v1[i], v2[i], v3[i])
    #get bounding box
    #get plane of polygon
    #get intersection points
    #construct 2 lines with interesction points
    #create voxel plane inside box
    #cut off voxel not inside polygon
    def voxelizeTriangle3(self, v1, v2, v3):
        minPoint = min()

    def voxelizeTriangle4(self, v1, v2, v3):
        edge1 = self.voxalizeEdge(v1,v2, self.getSteps(v1, v2))
        edge2 = self.voxalizeEdge(v1,v3, self.getSteps(v1, v3))
        edge3 = self.voxalizeEdge(v2,v3, self.getSteps(v2, v3))
        #print(f"{len(edge1)} {len(edge2)} {len(edge3)}")
        i = 0
        stop = len(edge1)

        for e2 in edge2:
            if i >= len(edge3):
                break
            delta = (e2[0] - edge1[0][0], e2[1] - edge1[0][1], e2[2] - edge1[0][2])
            #print(delta)
            #i += 1
            #if i == 3:
            #    return
            broken = False

            for j in range(stop):
                vec = (edge1[j][0] + delta[0], edge1[j][1] + delta[1], edge1[j][2] + delta[2])
                if vec == edge3[i]:
                    broken = True
                    stop = j+1
                    self.map.place(vec)
                    break
                if i < len(edge3)-1 and vec == edge3[i+1]:
                    i += 1
                    broken = True
                    stop = j+1
                    self.map.place(vec)
                    break

                if j == stop - 1:
                    break
                
                self.map.place(vec)

            #if not broken:
            #    print(e2)



    def voxelizeTriangle(self, v1, v2, v3):
        #voxel distance between each edge
        allSteps = sorted([(self.getSteps(v1,v2), v1, v2, v3), (self.getSteps(v1,v3), v1, v3, v2), (self.getSteps(v2,v3), v2, v3, v1)], key=lambda dist: dist[0])

        #get starting position
        if ((allSteps[0][1] == allSteps[1][1]).all() or (allSteps[0][1] == allSteps[1][2]).all()):
            x1 = allSteps[0][1]
            x2 = allSteps[0][2]
            x3 = allSteps[0][3]
        else:
            x1 = allSteps[0][2]
            x2 = allSteps[0][1]
            x3 = allSteps[0][3]

        totalStepsX = allSteps[0][0]
        totalStepsY = allSteps[1][0] * 2 # multiply by two to reduce artifacts on very smooth faces
        #totalStepsY = allSteps[1][0] * 2

        deltaX = (x2[0]-x1[0], x2[1]-x1[1], x2[2]-x1[2])
        deltaY = (x3[0]-x1[0], x3[1]-x1[1], x3[2]-x1[2])

        #s1 = len(tris) * totalStepsY

        #print(totalStepsX, totalStepsY, allSteps[2][0])
        #for stepY in range(3):
        for stepY in range(totalStepsY):
            #t1 = time.time()
            scal = stepY/totalStepsY
            startPoint = [x1[i] + (x3[i]-x1[i]) * scal for i in range(3)]
            endPoint = [x2[i] + (x3[i]-x2[i]) * scal for i in range(3)]
            totalSteps = self.getSteps(startPoint, endPoint)
            #totalSteps = self.getSteps((a,b,c), endPoint)
            #print(endPoint, (a,b,c), totalSteps)
            #print(startPoint, endPoint, totalSteps)
            #t2 = time.time()
            #totalStepTime += t2 - t1
            #s2 = stepY + totalStepsY * t
            #if s2 % 100 == 0:
            #    print(s1, s2)


            for stepX in range(totalSteps+1):
                # startpos + direction to x2 + direction to x3
                # each step get closer to target either in x,y or z direction.
                #t1 = time.time()
                relativeStepX = stepX/totalStepsX
                relativeStepY = stepY/totalStepsY


                a = x1[0] + deltaX[0]*relativeStepX + deltaY[0]*relativeStepY
                b = x1[1] + deltaX[1]*relativeStepX + deltaY[1]*relativeStepY
                c = x1[2] + deltaX[2]*relativeStepX + deltaY[2]*relativeStepY
                t2 = time.time()
                #totalCalcTime += t2 - t1

                # round to a voxel position
                x = self.roundFast(a)
                y = self.roundFast(b)
                z = self.roundFast(c)

                #x = round((a+1) * self.gridSize) - 1 
                #y = round((b+1) * self.gridSize) - 1 
                #z = round((c+1) * self.gridSize) - 1 

                #x = round(a, self.roundDevider)
                #y = round(b, self.roundDevider)
                #z = round(c, self.roundDevider)

                #t3 = time.time()
                #totalRoundTime += t3 - t2

                #print(a,b,c,x,y,z)
                    
                self.voxels[x*1000000+y*1000+z] = (x,y,z)
                #self.map.place((x,y,z))
                #totalDictTime += time.time() - t3


    def showVoxels(self):
        #self.voxs = np.array(list(self.map.convToCSC()), dtype=np.float32)
        self.voxs = np.array(list(self.voxels.values()), dtype=np.float32) / self.gridSize - 1
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




'''
def voxelize(self, verts, tris):
        #voxels = [[[False]*100 for i in range(100)] for j in range(100)]
        for t in range(0,1):
            tri = tris[t]
            v1  = verts[tri[0]]
            v2 = verts[tri[1]]
            v3 = verts[tri[2]]
            #print(x1,x2,x3)

            #voxel distance between each edge
            allSteps = sorted([(self.getSteps(v1,v2), v1, v2, v3), (self.getSteps(v1,v3), v1, v3, v2), (self.getSteps(v2,v3), v2, v3, v1)], key=lambda dist: dist[0])

            #get starting position
            if ((allSteps[0][1] == allSteps[1][1]).all() or (allSteps[0][1] == allSteps[1][2]).all()):
                x1 = allSteps[0][1]
                x2 = allSteps[0][2]
                x3 = allSteps[0][3]
            else:
                x1 = allSteps[0][2]
                x2 = allSteps[0][1]
                x3 = allSteps[0][3]

            totalStepsX = allSteps[0][0]
            totalStepsY = allSteps[1][0]

            print(totalStepsX, totalStepsY, allSteps[2][0])

            for stepY in range(totalStepsY):
                totalSteps = totalStepsX
                for stepX in range(0,totalSteps+1):
                    # startpos + direction to x2 + direction to x3
                    # each step get closer to target either in x,y or z direction.
                    a = x1[0] + (x2[0]-x1[0])/totalStepsX*stepX + (x3[0]-x1[0])/totalStepsY*stepY
                    b = x1[1] + (x2[1]-x1[1])/totalStepsX*stepX + (x3[1]-x1[1])/totalStepsY*stepY
                    c = x1[2] + (x2[2]-x1[2])/totalStepsX*stepX + (x3[2]-x1[2])/totalStepsY*stepY

                    # round to a voxel position
                    x = round(a,self.gridSize//10)
                    y = round(b,self.gridSize//10)
                    z = round(c,self.gridSize//10)

                    if stepX == 0:
                        scal = stepY/totalStepsY
                        startPoint = [x1[i] + (x3[i]-x1[i]) * scal for i in range(3)]
                        endPoint = [x2[i] + (x3[i]-x2[i]) * scal for i in range(3)]
                        totalSteps = self.getSteps(startPoint, endPoint)
                        #totalSteps = self.getSteps((a,b,c), endPoint)
                        print(endPoint, (a,b,c), totalSteps)
                        stepX += totalStepsX - totalSteps
                    
                    if totalSteps < stepX:
                        break
                        
                    self.voxels[x*100000000+y*10000+z] = (x,y,z)
        
        print(len(self.voxels))

'''