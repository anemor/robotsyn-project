import numpy as np

def rotate_x(radians): #Rotation about X-axis
    return np.array([[1, 0, 0, 0],
                    [0, np.cos(radians), -np.sin(radians), 0],
                    [0, np.sin(radians), np.cos(radians), 0],
                    [0, 0, 0, 1]])
def rotate_y(radians): #Rotation about Y-axis
    return np.array([[np.cos(radians), 0, np.sin(radians), 0],
                    [0, 1, 0, 0],
                    [-np.sin(radians), 0, np.cos(radians), 0],
                    [0,0,0,1]])
def rotate_z(radians): #Rotation about Z-axis
    return np.array([[np.cos(radians), -np.sin(radians), 0,0],
                    [np.sin(radians), np.cos(radians), 0,0],
                    [0, 0, 1,0],
                    [0,0,0,1]])
#
# For example:
def translate_x(x,y,z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])

