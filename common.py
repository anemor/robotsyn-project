import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

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

def project(K, X):
    """
    Computes the pinhole projection of a 3xN array of 3D points X
    using the camera intrinsic matrix K. Returns the dehomogenized
    pixel coordinates as an array of size 2xN.
    """

    # Tip: Use the @ operator for matrix multiplication, the *
    # operator on arrays performs element-wise multiplication!

    #
    # Placeholder code (replace with your implementation)
    #

    u_tilde=K@X[:3,:]
    N = len(X[0])
    u = np.zeros([2,N])
    for i in range(N):
        u[0][i]=u_tilde[0][i]/u_tilde[2][i]
        u[1][i]=u_tilde[1][i]/u_tilde[2][i]
        #u[2][i]=u_tilde[2][i]/u_tilde[3][i]
    return u

def draw_frame(K, T, scale=1, labels=False):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.

    Control the length of the axes by specifying the scale argument.
    """
    X = T @ np.array([
        [0,scale,0,0],
        [0,0,scale,0],
        [0,0,0,scale],
        [1,1,1,1]])
    u,v = project(K, X)
    plt.plot([u[0], u[1]], [v[0], v[1]], color='#cc4422') # X-axis
    plt.plot([u[0], u[2]], [v[0], v[2]], color='#11ff33') # Y-axis
    plt.plot([u[0], u[3]], [v[0], v[3]], color='#3366ff') # Z-axis
    if labels:
        textargs = {'color': 'w', 'va': 'center', 'ha': 'center', 'fontsize': 'x-small', 'path_effects': [PathEffects.withStroke(linewidth=1.5, foreground='k')]}
        plt.text(u[1], v[1], 'X', **textargs)
        plt.text(u[2], v[2], 'Y', **textargs)
        plt.text(u[3], v[3], 'Z', **textargs)
