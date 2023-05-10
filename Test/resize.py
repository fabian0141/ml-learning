from PIL import Image
from multiprocessing import Pool
import os

size = 128
sourcePath = "C:\\Users\\Fabian\\Documents\\AI_Development\\DataSets\\MegaScans\\Images\\"
folder = "img_align_plants"

def resizeImage(data):
    image = Image.open("C:\\Users\\Fabian\\Documents\\AI_Development\\DataSets\\MegaScans\\Images\\{}\\{}".format(folder, data)) 
    image = image.resize((size, size), Image.ANTIALIAS)
    im = image.convert("L")
    im.save("C:\\Users\\Fabian\\Documents\\AI_Development\\DataSets\\MegaScans\\Images{}\\{}\\{}_x{}.png".format(size, folder, data[:-4], size))

if __name__ == '__main__':

    filenames = os.listdir(sourcePath + folder)
    with Pool(16) as p:
        p.map(resizeImage, filenames)