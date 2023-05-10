import math
import numpy as np

def computeGCD(x, y, z):
    def gcd(x, y):
        if (x == 0):
            return x
        while(y and abs(y) >= 1e-12):
            x, y = y, x % y
        return abs(x)
    return gcd(gcd(x,y),z)

def getRotation3(rotation):
        angX = math.radians(rotation.x)
        angY = math.radians(rotation.y)
        angZ = math.radians(rotation.z)

        # Rotation around x-axis
        rot_x = np.array([[1.0, 0.0, 0.0],
                        [0.0, math.cos(angX), -math.sin(angX)],
                        [0.0, math.sin(angX), math.cos(angX)]], dtype=np.float32)

        # Rotation around y-axis
        rot_y = np.array([[math.cos(angY), 0.0, math.sin(angY)],
                        [0.0, 1.0, 0.0],
                        [-math.sin(angY), 0.0, math.cos(angY)]], dtype=np.float32)

        # Rotation around z-axis
        rot_z = np.array([[math.cos(angZ), -math.sin(angZ), 0.0],
                        [math.sin(angZ), math.cos(angZ), 0.0],
                        [0.0, 0.0, 1.0]], dtype=np.float32)

        # Combine the rotations into one matrix
        return rot_z @ rot_y @ rot_x

def getRotation4(rotation):
        angX = math.radians(rotation.x)
        angY = math.radians(rotation.y)
        angZ = math.radians(rotation.z)

        # Rotation around x-axis
        rot_x = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, math.cos(angX), -math.sin(angX), 0.0],
                        [0.0, math.sin(angX), math.cos(angX), 0.0],
                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        # Rotation around y-axis
        rot_y = np.array([[math.cos(angY), 0.0, math.sin(angY), 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [-math.sin(angY), 0.0, math.cos(angY), 0.0],
                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        # Rotation around z-axis
        rot_z = np.array([[math.cos(angZ), -math.sin(angZ), 0.0, 0.0],
                        [math.sin(angZ), math.cos(angZ), 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        # Combine the rotations into one matrix
        return rot_z @ rot_y @ rot_x