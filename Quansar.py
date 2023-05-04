import matplotlib.pyplot as plt
import numpy as np
from common import *

class Quanser:
    def __init__(self):
        self.K = np.loadtxt('./data/calibration/K.txt')
        #self.heli_points = np.loadtxt('./data/heli_points.txt').T må legge inn
        self.base_0=np.array([0.0,0.0,0.0,1.0])
        self.X_p = np.c_[self.base_0].reshape(4, 1)

        #self.platform_to_camera = np.loadtxt('./data/platform_to_camera.txt') må legge inn
        self.transform_camera_base_point=translate_x(0.03,-0.1,-0.52)@rotate_z(np.deg2rad(-3.25))@rotate_y(np.deg2rad(-90))@rotate_z(np.deg2rad(90))

    def residuals(self, u, weights, p):
        #wrist_y=p[0] 
        #wrist_z=p[1]
        #first_y=p[2]
        #first_z=p[3]
        #second_z=p[4]
        #third_z=p[5]
        # Compute the helicopter coordinate frames
        wrist=rotate_y(p[0])@rotate_z(p[1])

        index_wrist_to_first=translate_x(0.083,0,0.028)@rotate_y(p[2])@rotate_z(p[3])
        index_first_to_second=translate_x(0.035,0,0)@rotate_z(p[4])
        index_second_to_third=translate_x(0.025,0.0,0)@rotate_z(p[5])
        index_third_to_fingertip=translate_x(0.0245,0.00,0)

        middle_wrist_to_first=translate_x(0.09,0,0.0065)@rotate_y(p[6])@rotate_z(p[7])
        middle_first_to_second=translate_x(0.035,0,0)@rotate_z(p[8])
        middle_second_to_third=translate_x(0.03,0.0,0)@rotate_z(p[9])
        middle_third_to_fingertip=translate_x(0.028,0.00,0)

        ring_wrist_to_first=translate_x(0.08,0,-0.01)@rotate_y(p[10])@rotate_z(p[11])
        ring_first_to_second=translate_x(0.036,0,0)@rotate_z(p[12])
        ring_second_to_third=translate_x(0.027,0.0,0)@rotate_z(p[13])
        ring_third_to_fingertip=translate_x(0.027,0.00,0)

        little_wrist_to_first=translate_x(0.07,0,-0.028)@rotate_y(p[14])@rotate_z(p[15])
        little_first_to_second=translate_x(0.032,0,0)@rotate_z(p[16])
        little_second_to_third=translate_x(0.021,0.0,0)@rotate_z(p[17])
        little_third_to_fingertip=translate_x(0.0235,0.00,0)

        self.wrist_to_camera=self.transform_camera_base_point@wrist

        self.index_first_to_camera=self.wrist_to_camera@index_wrist_to_first
        self.index_second_to_camera=self.index_first_to_camera@index_first_to_second
        self.index_third_to_camera=self.index_second_to_camera@index_second_to_third
        self.index_fingertip_to_camera=self.index_third_to_camera@index_third_to_fingertip

        self.middle_first_to_camera=self.wrist_to_camera@middle_wrist_to_first
        self.middle_second_to_camera=self.middle_first_to_camera@middle_first_to_second
        self.middle_third_to_camera=self.middle_second_to_camera@middle_second_to_third
        self.middle_fingertip_to_camera=self.middle_third_to_camera@middle_third_to_fingertip

        self.ring_first_to_camera=self.wrist_to_camera@ring_wrist_to_first
        self.ring_second_to_camera=self.ring_first_to_camera@ring_first_to_second
        self.ring_third_to_camera=self.ring_second_to_camera@ring_second_to_third
        self.ring_fingertip_to_camera=self.ring_third_to_camera@ring_third_to_fingertip

        self.little_first_to_camera=self.wrist_to_camera@little_wrist_to_first
        self.little_second_to_camera=self.little_first_to_camera@little_first_to_second
        self.little_third_to_camera=self.little_second_to_camera@little_second_to_third
        self.little_fingertip_to_camera=self.little_third_to_camera@little_third_to_fingertip
        

        #base_to_platform = translate_x(0.1145/2, 0.1145/2, 0.0)@rotate_z(yaw)
        #hinge_to_base    = translate_x(0.00, 0.00,  0.325)@rotate_y(pitch)
        #arm_to_hinge     = translate_x(0.00, 0.00, -0.050)
        #rotors_to_arm    = translate_x(0.65, 0.00, -0.030)@rotate_x(roll)
        #self.base_to_camera   = self.platform_to_camera@base_to_platform
        #self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        #self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        #self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        # Compute the predicted image location of the markers, kommer til å ha 5? p-er en for hvert punkt
        p0=self.wrist_to_camera@self.X_p
        p1=self.index_first_to_camera@self.X_p
        p2=self.index_second_to_camera@self.X_p
        p3=self.index_third_to_camera@self.X_p
        p4=self.index_fingertip_to_camera@self.X_p

        p5=self.middle_first_to_camera@self.X_p
        p6=self.middle_second_to_camera@self.X_p
        p7=self.middle_third_to_camera@self.X_p
        p8=self.middle_fingertip_to_camera@self.X_p

        p9=self.ring_first_to_camera@self.X_p
        p10=self.ring_second_to_camera@self.X_p
        p11=self.ring_third_to_camera@self.X_p
        p12=self.ring_fingertip_to_camera@self.X_p

        p13=self.little_first_to_camera@self.X_p
        p14=self.little_second_to_camera@self.X_p
        p15=self.little_third_to_camera@self.X_p
        p16=self.little_fingertip_to_camera@self.X_p
        #p1 = self.arm_to_camera @ self.heli_points[:,:3]
        #p2 = self.rotors_to_camera @ self.heli_points[:,3:]
        hat_u = project(self.K, np.hstack([p0, p1, p2, p3, p4, p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16]))
        self.hat_u = hat_u # Save for use in draw()

        r=hat_u-u
        #r=np.reshape(r,2*7)
        #tror shape nå blir 2*17
        r[0]=r[0]*weights
        r[1]=r[1]*weights
        r=np.hstack(r)
        #
        # TASK: Compute the vector of residuals.
        #
        # Tip: Use np.hstack to concatenate the horizontal and vertical residual components
        # into a single 1D array. Note: The plotting code will not work correctly if you use
        # a different ordering.
        #r = np.zeros(2*7) # Placeholder, remove me!
        return r

    def draw(self, u, weights, image_number):
        #I = plt.imread('./data_vid/data/video%04d.jpg' % image_number)
        I=plt.imread('./data/video-bilder/image34.jpg')
        plt.imshow(I)
        plt.scatter(*u[:, weights == 1], linewidths=1, edgecolor='black', color='white', s=60, label='Observed')
        plt.scatter(*self.hat_u, color='red', label='Predicted', s=10)
        plt.legend()
        plt.title('Reprojected frames and points on image number %d' % image_number)
        draw_frame(self.K, self.transform_camera_base_point, scale=0.025)
        draw_frame(self.K, self.wrist_to_camera, scale=0.025)
        draw_frame(self.K, self.index_first_to_camera, scale=0.025)
        draw_frame(self.K, self.index_second_to_camera, scale=0.025)
        draw_frame(self.K, self.index_third_to_camera, scale=0.025)
        draw_frame(self.K, self.index_fingertip_to_camera, scale=0.025)
        
        plt.xlim([0, I.shape[1]])
        plt.ylim([I.shape[0], 0])
