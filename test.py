from Test.cave import testcave
from Graphics import graphics, visualizevoxels
from Cuda import voxelizer, testviews

if __name__ == '__main__':
    runProgram = 1

    match runProgram:
        case 0:
            testcave.run()
        case 1:
            graphics.run()
        case 2:
            voxelizer.run()
            visualizevoxels.run("..\\DataSets\\MegaScans\\Voxel128", 3000)
        case 3:
            visualizevoxels.run("..\\DataSets\\MegaScans\\Voxel64", 7115) #6436
        case 4:
            testviews.run()

