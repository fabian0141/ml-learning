from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader
import numpy as np
from Graphics.helper.mathext import computeGCD
import glm
import time
import math

class Voxelize:
    def __init__(self, gridSize):
        self.gridSize = gridSize
        self.voxels = {}
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

    def voxalizeEdge(self, p1, p2, totalSteps):
        edge = []
        for stepX in range(totalSteps+1):
            a = p1[0] + (p2[0]-p1[0])/totalSteps*stepX
            b = p1[1] + (p2[1]-p1[1])/totalSteps*stepX
            c = p1[2] + (p2[2]-p1[2])/totalSteps*stepX


            x =round(a,self.gridSize//10)
            y =round(b,self.gridSize//10)
            z =round(c,self.gridSize//10)
            self.voxels[x*1000000+y*1000+z] = (x,y,z)
        return edge

    def roundFast(self, num):
        return int((num+2) * self.gridSize + 0.5) / self.gridSize - 2


    def voxelize(self, verts, tris):
        #totalDictTime = 0
        #totalStepTime = 0
        #totalRoundTime = 0
        #totalCalcTime = 0
        start = time.time()
        #voxels = [[[False]*100 for i in range(100)] for j in range(100)]

        if tris.size == 0:
            for i in range(0, len(verts), 3):
                self.voxelizeTriangle(verts[i],verts[i+1],verts[i+2])
        else:
            #for t in range(len(tris)):
            for t in range(0, 1):
                tri = tris[t]
                v1  = verts[tri[0]]
                v2 = verts[tri[1]]
                v3 = verts[tri[2]]
                #print(x1,x2,x3)
                #self.voxalizeEdge(v1,v2, self.getSteps(v1, v2))
                #self.voxalizeEdge(v2,v3, self.getSteps(v2, v3))
                #self.voxalizeEdge(v1,v3, self.getSteps(v1, v3))


                self.voxelizeTriangle(v1,v2,v3)

        end = time.time()
        #print(f'Voxel time: {end - start} totalStepTime: {totalStepTime} totalCalcTime: {totalCalcTime} totalRoundTime: {totalRoundTime} totalDictTime: {totalDictTime}')
        print(f'Voxel time: {end - start}')
        print(len(self.voxels))


    def getCornerPoints(vec1, vec2, size, axis):
        
        for




        if v1[axis] < v2[axis]:
            if v1[axis] 

    def voxelizeTriangle2(self, v1, v2, v3):

        #get surface by two vectors and 
        vec1 = (v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2])
        vec2 = (v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2])


        xP = 
        yP
        zP

        minPoint = [min(v1[0], v2[0], v3[0]), min(v1[1], v2[1], v3[1]), min(v1[2], v2[2], v3[2])]
        maxPoint = [max(v1[0], v2[0], v3[0]), max(v1[1], v2[1], v3[1]), max(v1[2], v2[2], v3[2])]

        

        p1 = [v1[0], v1[1], ]

        p2 = [minPoint[0], maxPoint[1], maxPoint[2]]
        p3 = [maxPoint[0], minPoint[1], maxPoint[2]]

        stepsX = getSteps()
        stepsY = 
        a = p1[0] + (maxPoint[0]-minPoint[0])/totalSteps*stepX
            b = p1[1] + (p2[1]-p1[1])/totalSteps*stepX
            c = p1[2] + (p2[2]-p1[2])/totalSteps*stepX


            x =round(a,self.gridSize//10)
            y =round(b,self.gridSize//10)
            z =round(c,self.gridSize//10)
            self.voxels[x*1000000+y*1000+z] = (x,y,z)

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
        totalStepsY = allSteps[1][0] # multiply by two to reduce artifacts on very smooth faces
        #totalStepsY = allSteps[1][0] * 2

        deltaX = (x2[0]-x1[0], x2[1]-x1[1], x2[2]-x1[2])
        deltaY = (x3[0]-x1[0], x3[1]-x1[1], x3[2]-x1[2])

        #s1 = len(tris) * totalStepsY

        #print(totalStepsX, totalStepsY, allSteps[2][0])
        for stepY in range(3):
        #for stepY in range(totalStepsY):
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

                #x = round(a, self.roundDevider)
                #y = round(b, self.roundDevider)
                #z = round(c, self.roundDevider)

                #t3 = time.time()
                #totalRoundTime += t3 - t2

                #print(a,b,c,x,y,z)
                    
                self.voxels[x*1000000+y*1000+z] = (x,y,z)
                #totalDictTime += time.time() - t3


    def showVoxels(self):
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
        glDrawArraysInstanced(GL_TRIANGLES, 0, 36, len(self.voxels))
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