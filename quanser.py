import matplotlib.pyplot as plt
import numpy as np
from common import *

class Quanser:
    def __init__(self):
        self.K = np.loadtxt('data/K.txt')
        self.heli_points = np.loadtxt('data/heli_points.txt').T
        self.platform_to_camera = np.loadtxt('data/platform_to_camera.txt')

    def residuals(self, u, weights, yaw, pitch, roll):
        # Compute the helicopter coordinate frames
        base_to_platform = translate(0.1145/2, 0.1145/2, 0.0)@rotate_z(yaw)
        hinge_to_base    = translate(0.00, 0.00,  0.325)@rotate_y(pitch)
        arm_to_hinge     = translate(0.00, 0.00, -0.050)
        rotors_to_arm    = translate(0.65, 0.00, -0.030)@rotate_x(roll)
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        # Compute the predicted image location of the markers
        p1 = self.arm_to_camera @ self.heli_points[:,:3] # første 3 linjene/arm markers
        p2 = self.rotors_to_camera @ self.heli_points[:,3:] # resterende 4 linjene/rotor markers
        hat_u = project(self.K, np.hstack([p1, p2]))
        self.hat_u = hat_u # Save for use in draw()
        
        #
        # TASK: Compute the vector of residuals.
        #
        # Tip: Use np.hstack to concatenate the horizontal and vertical residual components
        # into a single 1D array. Note: The plotting code will not work correctly if you use
        # a different ordering.
        
        # residual is difference in horizontal and vertical
        u_weighted = u * np.array([weights, weights])
        u_hat_weighted = u * np.array([weights, weights])

        u1 = u_weighted[:1]
        u2 = u_weighted[1:]
        u_hat1 = u_hat_weighted[:1]
        u_hat2 = u_hat_weighted[1:]
        #r = np.hstack((horizontal, vertical))
        r = np.reshape(np.hstack((u_hat1-u1, u_hat2-u2)), -1)
        #r = np.hstack((hat_u[:1], hat_u[1:]))
        #r = np.zeros(2*7) # Placeholder, remove me!

        return r

    def draw(self, u, weights, image_number):
        I = plt.imread('data/video%04d.jpg' % image_number)
        plt.imshow(I)
        plt.scatter(*u[:, weights == 1], linewidths=1, edgecolor='black', color='white', s=80, label='Observed')
        plt.scatter(*self.hat_u, color='red', label='Predicted', s=10)
        plt.legend()
        plt.title('Reprojected frames and points on image number %d' % image_number)
        draw_frame(self.K, self.platform_to_camera, scale=0.05)
        draw_frame(self.K, self.base_to_camera, scale=0.05)
        draw_frame(self.K, self.hinge_to_camera, scale=0.05)
        draw_frame(self.K, self.arm_to_camera, scale=0.05)
        draw_frame(self.K, self.rotors_to_camera, scale=0.05)
        plt.xlim([0, I.shape[1]])
        plt.ylim([I.shape[0], 0])
