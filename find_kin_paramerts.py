import matplotlib.pyplot as plt
import numpy as np
from Model import *
from scipy.optimize import least_squares
from plot_all import plot_all

#subset = np.arange(1, 350, 20)

all_detections = np.loadtxt('./data/detections_correct.txt')
#all_detections = all_detections[subset, :]
all_weights = all_detections[:, ::3]
all_u       = all_detections[:, 1::3]
all_v       = all_detections[:, 2::3]

model = ModelA()
# model_name, model = 'B', ModelB()

        #index_wrist_to_first=translate_x(0.083,0,0.028)@rotate_y(p[2])@rotate_z(p[3])
        #index_first_to_second=translate_x(0.035,0,0)@rotate_z(p[4])
        #index_second_to_third=translate_x(0.025,0.0,0)@rotate_z(p[5])
        #index_third_to_fingertip=translate_x(0.0245,0.00,0)

init_p = []
#if model_name == 'A':
init_p.extend([0.083, 0.028, 0.035, 0.025, 0.0245])
base_1=np.array([0.0,0.0,0.0])
X_p = np.c_[base_1,base_1,base_1,base_1,base_1].reshape(-1)

#init_p.extend(np.loadtxt('../data/heli_points.txt')[:,:3].reshape(-1))
init_p.extend(X_p)
#elif model_name == 'B':
#    init_p.extend([0.0, 0.0, 0.0, 0.1145/2, 0.1145/2, 0.0])
#    init_p.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.325])
#    init_p.extend([0.0, 0.0, 0.0, 0.65, 0.0, -0.08])
#     init_points = np.loadtxt('../data/heli_points.txt')
    # Subtracting 0.05 from the Z coordinates of the arm markers
    # is only if you want the initialization of Model B to give
    # the exact same transformations as the initialization of A.
#    init_points[:3,2] -= 0.05
#    init_p.extend(init_points[:,:3].reshape(-1))
init_p.extend(np.loadtxt('trajectory.txt').reshape(-1))
init_p = np.array(init_p)

nimg = all_u.shape[0]   # Number of images
rdim = all_u.shape[1]*2 # Number of residuals in one image
sdim = 6                # Number of state parameters in one image
kdim = len(init_p) - nimg*sdim # Number of kinematic parameters

resfun = lambda p : model.residuals_for_all(all_u, all_v, all_weights, p[:kdim], p[kdim:])

p = least_squares(fun=resfun, x0=init_p, max_nfev=50, verbose=2).x

# Estimate the state on all images (not just the subset).
# The code below is mostly identical to part1b.
# print(p)
model.set_kinematic_parameters(p[:kdim])
all_detections = np.loadtxt('./data/detections_correct.txt')
all_r = []
all_p = []
p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
for i in range(len(all_detections)):
    weights = all_detections[i, ::3]
    u = np.vstack((all_detections[i, 1::3], all_detections[i, 2::3]))
    resfun = lambda p : model.residuals_for_one(u, weights, p)
    p = least_squares(resfun, x0=p, method='lm').x
    all_r.append(resfun(p))
    all_p.append(p)
all_p = np.array(all_p)
all_r = np.array(all_r)
plot_all(all_p, all_r, all_detections, subtract_initial_offset=True)
plt.savefig(f'our_model_kinematics.png')
plt.show()