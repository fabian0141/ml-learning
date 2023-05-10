from OpenGL.GL import glEnableClientState, GL_VERTEX_ARRAY
import time
from Test.cave.cagen import CellularCaveGenerator
from Graphics.models.cube import TestCube
from Graphics.models.objloader import ObjLoader
from Graphics.controls.game import Game
from Graphics.controls.camera import Camera

def run():
    # Initialize Pygame and OpenGL
    game = Game()
    camera = Camera(Camera.PERSPECTIVE_VIEW)
    cagen = CellularCaveGenerator((64, 32, 64))

    start = time.time()
    cagen.generateCave()
    end = time.time()
    print(f'Voxel time {time.time() - start}')

    cagen.showCave()
    glEnableClientState(GL_VERTEX_ARRAY)

    # Render the OBJ file
    while True:
        game.loopBeginning()
        camera.checkControls(speed=30)

        (projection, view, model) = camera.getWorldView()

        #obj.renderObj(camera.transform, projection, view, model)
        #tcube.renderCube(camera.transform, projection, view, model)
        cagen.renderCave(camera.transform, projection, view, model)
        
        game.loopEnd(60)