import matplotlib.pyplot as plt
import numpy as np
from common import *

class BaseModel:
    def __init__(self):
        self.K                  = np.loadtxt('./data/calibration/K.txt')
        #self.platform_to_camera = np.loadtxt('../data/platform_to_camera.txt')
        self.transform_camera_base_point=translate_x(0.03,-0.1,-0.52)@rotate_z(np.deg2rad(-3.25))@rotate_y(np.deg2rad(-90))@rotate_z(np.deg2rad(90))

    def residuals_for_all(self, all_u, all_v, all_weights, p_kinematic, all_p_state):
        self.set_kinematic_parameters(p_kinematic)
        nimg = all_u.shape[0]
        rdim = all_u.shape[1]*2
        all_r = np.empty(nimg*rdim)
        for i in range(nimg):
            u = np.vstack((all_u[i], all_v[i]))
            weights = all_weights[i, :]
            p_state = all_p_state[3*i:3*i+3]
            r = self.residuals_for_one(u, weights, p_state)
            all_r[i*rdim:i*rdim+rdim] = r
        return all_r

class ModelA(BaseModel):
    def set_kinematic_parameters(self, p_kinematic):
        l1, l2, l3, l4, l5, \
        X1, Y1, Z1, \
        X2, Y2, Z2, \
        X3, Y3, Z3, \
        X4, Y4, Z4, \
        X5, Y5, Z5 = p_kinematic

        self.T1 = translate_x(l1, 0.0, l2)
        self.T2 = translate_x(l3, 0.0, 0.0)
        self.T3 = translate_x(l4, 0.0, 0.0)
        self.T4 = translate_x(l5, 0.0, 0.0)
        self.wrist = np.array([
            [X1],
            [Y1],
            [Z1],
            [1]])
        self.first = np.array([
            [X2],
            [Y2],
            [Z2],
            [1]])
        self.second = np.array([
            [X3],
            [Y3],
            [Z3],
            [1]])
        self.third = np.array([
            [X4],
            [Y4],
            [Z4],
            [1]])
        self.fingertip = np.array([
            [X5],
            [Y5],
            [Z5],
            [1]])

    def residuals_for_one(self, u, weights, p_state):
        
        #base_to_platform = self.T1@rotate_z(p_state[0])
        #hinge_to_base    = self.T2@rotate_y(p_state[1])
        #arm_to_hinge     = self.T3
        #rotors_to_arm    = self.T4@rotate_x(p_state[2])
        wrist=rotate_y(p_state[0])@rotate_z(p_state[1])

        index_wrist_to_first=self.T1@rotate_y(p_state[2])@rotate_z(p_state[3])
        index_first_to_second=self.T2@rotate_z(p_state[4])
        index_second_to_third=self.T3@rotate_z(p_state[5])
        index_third_to_fingertip=self.T4

        #self.base_to_camera   = self.platform_to_camera@base_to_platform
        #self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        #self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        #self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        self.wrist_to_camera=self.transform_camera_base_point@wrist

        self.index_first_to_camera=self.wrist_to_camera@index_wrist_to_first
        self.index_second_to_camera=self.index_first_to_camera@index_first_to_second
        self.index_third_to_camera=self.index_second_to_camera@index_second_to_third
        self.index_fingertip_to_camera=self.index_third_to_camera@index_third_to_fingertip

        #hat_u = self.K @ np.hstack([
        #    self.arm_to_camera @ self.points_in_arm,
        #    self.rotors_to_camera @ self.points_in_rotors
        #])[:3,:]

        hat_u = self.K @ np.hstack([
            self.wrist_to_camera @ self.wrist,
            self.index_first_to_camera @ self.first,
            self.index_second_to_camera @ self.second,
            self.index_third_to_camera@self.third,
            self.index_fingertip_to_camera@self.fingertip
        ])[:3,:]

        hat_u = hat_u[:2,:]/hat_u[2,:]
        r = weights*(hat_u - u)
        return np.hstack([r[0,:], r[1,:]])