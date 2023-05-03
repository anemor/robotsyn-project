import matplotlib.pyplot as plt
import numpy as np
from gauss_newton import gauss_newton
from quanser import Quanser

# Image to load data for (must be in [0, 350])
image_number = 0

# Tip:
# "u" is a 2x7 array of detected marker locations.
# It is the same size in every image, but some of its
# entries may be invalid if the corresponding markers were
# not detected. Which entries are valid is encoded in
# the "weights" array, which is a 1D array of length 7.
detections = np.loadtxt('data/detections.txt')
weights = detections[image_number, ::3]
u = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))

# weights[i] hører til u[0:i] og u[1:i] 
# der u[0:] har x og u[1:] har y (eller motsatt) 
# for hver av de 7? marker locationsa, i = 0,..,6
quanser = Quanser()

# Tip:
# Many optimization packages for Python expect you to provide a
# callable function that computes the residuals, and optionally
# the Jacobian, at a given parameter vector. The provided Gauss-Newton
# implementation also follows this practice. However, because the
# "residuals" method takes arguments other than the parameters, you
# must first define a "lambda function wrapper" that takes only a
# single argument (the parameter vector), and likewise for computing
# the Jacobian. This can be done like this:
resfun = lambda p : quanser.residuals(u, weights, p[0], p[1], p[2])
# resfun(p) vil sende p-verdien inn til lambdaen over
#resfun = lambda p : np.reshape(quanser.residuals(u, weights, p[0], p[1], p[2]), -1)
# Tip:
# These parameter values (yaw, pitch, roll) are close to the optimal
# estimates for image 0.
p = np.array([11.6, 28.9, 0.0])*np.pi/180
#p = np.array([11.6, 28.9, 0.0])*np.pi/180
p_low = np.array([0.0001, 0.0001, 0.0001])
p_high = np.array([100, 100, 100])*np.pi/180
#
# Task: Call gauss_newton  
#

# lage ny p
#p = gauss_newton(resfun, p0, step_size=0.9, num_steps=10)

# 1.4
p_opt = gauss_newton(resfun, p, step_size=0.9, num_steps=10)#, xtol=0.001)

#gn = lambda p : gauss_newton(resfun, p, step_size=0.9, num_steps=10)

r = resfun(p_opt)
# r = resfun(p)
print('Residuals on image %d:' % image_number)
print(r)

# Tip:
# This snippet produces the requested outputs for the second half
# of Task 1.3. The figure is 5saved to your working directory.
reprojection_errors = np.linalg.norm(r.reshape((2,-1)), axis=0)
print('Reprojection errors:')
for i,reprojection_error in enumerate(reprojection_errors):
    print('Marker %d: %5.02f px' % (i + 1, reprojection_error))
print('Average:  %5.02f px' % np.mean(reprojection_errors))
print('Median:   %5.02f px' % np.median(reprojection_errors))
quanser.draw(u, weights, image_number)
plt.savefig('out_part1a.png')
plt.show()
