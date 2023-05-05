import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from Quansar import Quanser, Quanser_Index
from plot_all import * #burde laste inn

detections = np.loadtxt('./data/detections_correct.txt')
# quanser=Quanser()
quanser=Quanser_Index()

p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#p= np.zeros(22)
all_r = []
all_p = []

#image_number =1
#step_size=0.9
#steps_nb=40

# Tip:
# "u" is a 2x7 array of detected marker locations.
# It is the same size in every image, but some of its
# entries may be invalid if the corresponding markers were
# not detected. Which entries are valid is encoded in
# the "weights" array, which is a 1D array of length 7.
#detections = np.loadtxt('./data/detections.txt')
#weights = detections[image_number, ::3]
#u = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))

#for i in range(len(detections)):
for i in range(len(detections)):
    weights = detections[i, ::3]
    u = np.vstack((detections[i, 1::3], detections[i, 2::3]))

# Tip: Lambda functions can be defined inside a for-loop, defining
# a different function in each iteration. Here we pass in the current
# image's "u" and "weights".
    resfun = lambda p : quanser.residuals(u, weights, p)

# Tip: Use the previous image's parameter estimate as initialization
    p = least_squares(resfun, x0=p, method='lm').x

# Collect residuals and parameter estimate for plotting later
    all_r.append(resfun(p))
    all_p.append(p)
all_p = np.array(all_p)
all_r = np.array(all_r)

#print(np.rad2deg(p))
#r = resfun(p)
#print('Residuals on image %d:' % image_number)
#print(r)

# Tip:
# This snippet produces the requested outputs for the second half
# of Task 1.3. The figure is saved to your working directory.
#reprojection_errors = np.linalg.norm(r.reshape((2,-1)), axis=0)
#print('Reprojection errors:')
#for i,reprojection_error in enumerate(reprojection_errors):
#    print('Marker %d: %5.02f px' % (i + 1, reprojection_error))
#print('Average:  %5.02f px' % np.mean(reprojection_errors))
#print('Median:   %5.02f px' % np.median(reprojection_errors))
#quanser.draw(u, weights, image_number)
#plt.savefig('out_part1a.png')
#plt.show()

print(np.rad2deg(all_p))

# Tip: This saves the estimated angles to a txt file.
# This can be useful for Part 3.
np.savetxt('trajectory.txt', all_p)
# It can be loaded as
# all_p = np.loadtxt('trajectory_from_part1.txt')

# Tip: See comment in plot_all.py regarding the last argument.
#plot_all(all_p, all_r, detections, subtract_initial_offset=True)
#plt.savefig('out_part1b.png')
#plt.show()