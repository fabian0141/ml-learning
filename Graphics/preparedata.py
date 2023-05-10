from OpenGL.GL import glEnableClientState, GL_VERTEX_ARRAY
from models.objloader import ObjLoader
from controls.game import Game
from controls.camera import Camera
import os
import glm
from OpenGL.GL import *
from PIL import Image
from multiprocessing import Pool
import threading

path = "C:\\Users\\Fabian\\Documents\\AI_Development\\DataSets\\MegaScans\\OBJ"

def loadObj(name):
    print(name)
    return ObjLoader(path + "\\" + name)

def saveImage(data, name):
    idx = 0
    for d in data:
        image = Image.frombytes(mode="RGB", size=(1024, 1024), data=d)
        image.rotate(180).save("C:\\Users\\Fabian\\Documents\\AI_Development\\DataSets\\MegaScans\\Images\\{}_{}.png".format(name, idx))
        idx += 1

if __name__ == '__main__':
    # Initialize Pygame and OpenGL
    game = Game(1024, 1024)
    camera = Camera()

    objs = []
    filenames = os.listdir(path)
    print(filenames)

    with Pool(16) as p:
        objs = p.map(loadObj, filenames)

    #for name in filenames:
    #    objs.append(ObjLoader(path + "\\" + name))
    #    print(name)
    #    break

    glEnableClientState(GL_VERTEX_ARRAY)

    icosphere = ObjLoader("C:\\Users\\Fabian\\Documents\\AI_Development\\DataSets\\icospheresmall.obj")
    views = icosphere.points

    print("Start rendering")

    # Render the OBJ file
    i = 0
    for obj in objs:
        print("Rendering ", filenames[i])
        j = 0
        obj.prepareRender()
        data = []

        for viewDir in views:
            viewVec = glm.vec3(viewDir) * 2
            if (abs(viewVec.y) > 0.999):
                viewVec.z = 0.0000000000001

            game.loopBeginning()
            camera.checkControls()

            (projection, view, model) = Camera.getWorldView(camera.translation, 1, viewVec)

            obj.renderObj(camera.transform, projection, view, model, viewVec)

            #data = glReadPixels(0, 0, 1024, 1024, GL_RGB, GL_UNSIGNED_BYTE)
            #image = Image.frombytes(mode="RGB", size=(1024, 1024), data=data)
            #image.save("C:\\Users\\Fabian\\Documents\\AI_Development\\DataSets\\MegaScans\\Images\\{}_{}.png".format(filenames[i], j))

            data.append(glReadPixels(0, 0, 1024, 1024, GL_RGB, GL_UNSIGNED_BYTE))
            #p = Process(target=saveImage, args=(data, filenames[i], j,))
            #p.start()

            game.loopEnd()
            j += 1
        thread = threading.Thread(target=saveImage, args=(data, filenames[i],))
        thread.start()
        i += 1
