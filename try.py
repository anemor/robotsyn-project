import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from common import *

#transform_camera_base_point=translate_x(0.03,-0.1,-0.52)@rotate_z(np.deg2rad(-3.25))@rotate_y(np.deg2rad(-90))@rotate_z(np.deg2rad(90))
transform_camera_base_point=translate_x(0.0,-0.08,-0.52)@rotate_y(np.deg2rad(-90))@rotate_z(np.deg2rad(90))
print(transform_camera_base_point)

K = np.loadtxt('./data/calibration/K.txt')

base_1=np.array([0.0,0.0,0.0,1.0])
#base_2=[0.1145,0.0,0.0,1.0]

X_p = np.c_[base_1].reshape(4, 1)

X_c = transform_camera_base_point@ X_p

u,v=project(K,X_c)

#Task 4.3
theta_0=0#np.deg2rad(0)
alpha=0#np.deg2rad(0) #denne har negative rotajon
index_wrist_to_first=transform_camera_base_point@translate_x(0.083,0,0.028)#@rotate_z(theta_0)@rotate_y(alpha)
middle_wrist_to_first=transform_camera_base_point@translate_x(0.09,0,0.0065)
ring_wrist_to_first=transform_camera_base_point@translate_x(0.08,0,-0.01)
little_wrist_to_first=transform_camera_base_point@translate_x(0.07,0,-0.028)
thumb_wrist_to_first=transform_camera_base_point@translate_x(0.007,0,0.028)@rotate_x(np.deg2rad(90))

#Task 4.4
theta_2=np.deg2rad(0)
index_first_to_second=index_wrist_to_first@translate_x(0.035,0,0)
middle_first_to_second=middle_wrist_to_first@translate_x(0.035,0,0)
ring_first_to_second=ring_wrist_to_first@translate_x(0.036,0,0)
little_first_to_second=little_wrist_to_first@translate_x(0.032,0,0)
thumb_first_to_second=thumb_wrist_to_first@translate_x(0.04,0.0225,0)
#base_to_platform@translate_x(0.0,0.0,0.325)@rotate_y(pitch)

#Task 4.5
theta_3=np.deg2rad(0)
index_second_to_third=index_first_to_second@translate_x(0.025,0.0,0)
middle_second_to_third=middle_first_to_second@translate_x(0.03,0.0,0)
ring_second_to_third=ring_first_to_second@translate_x(0.027,0.0,0)
little_second_to_third=little_first_to_second@translate_x(0.021,0.0,0)
thumb_second_to_third=thumb_first_to_second@translate_x(0.031,0,0)@rotate_z(np.deg2rad(-15))

#Task 4.6
theta_4=np.deg2rad(0)
index_third_to_fingertip=index_second_to_third@translate_x(0.0245,0.00,0)
middle_third_to_fingertip=middle_second_to_third@translate_x(0.028,0.00,0)
ring_third_to_fingertip=ring_second_to_third@translate_x(0.027,0.00,0)
little_third_to_fingertip=little_second_to_third@translate_x(0.0235,0.00,0)
thumb_third_to_fingertip=thumb_second_to_third@translate_x(0.03,0,0)

base_0=np.array([0.0,0.0,0.0,1.0])
base_first_middle=np.array([-0.006,0.0,0.0,1.0])
base_first_ring=np.array([-0.007,0.0,0.0,1.0])
X_p = np.c_[base_0].reshape(4, 1)
X_first_middle=np.c_[base_first_middle].reshape(4, 1)
X_first_ring=np.c_[base_first_ring].reshape(4, 1)

index_first = index_wrist_to_first@ X_p
index_second=index_first_to_second@X_p
index_third=index_second_to_third@X_p
index_fingertip=index_third_to_fingertip@X_p

middle_first = middle_wrist_to_first@ X_p
middle_second=middle_first_to_second@X_p
middle_third=middle_second_to_third@X_p
middle_fingertip=middle_third_to_fingertip@X_p

ring_first = ring_wrist_to_first@ X_p
ring_second=ring_first_to_second@X_p
ring_third=ring_second_to_third@X_p
ring_fingertip=ring_third_to_fingertip@X_p

little_first = little_wrist_to_first@ X_p
little_second=little_first_to_second@X_p
little_third=little_second_to_third@X_p
little_fingertip=little_third_to_fingertip@X_p

thumb_first=thumb_wrist_to_first@X_p
thumb_second=thumb_first_to_second@X_p
thumb_third=thumb_second_to_third@X_p
thumb_fingertip=thumb_third_to_fingertip@X_p

u_index_first, v_index_first=project(K,index_first)
u_index_second, v_index_second=project(K,index_second)
u_index_third, v_index_third=project(K,index_third)
u_index_finegrtip, v_index_fingertip=project(K,index_fingertip)

u_middle_first, v_middle_first=project(K,middle_first)
u_middle_second, v_middle_second=project(K,middle_second)
u_middle_third, v_middle_third=project(K,middle_third)
u_middle_finegrtip, v_middle_fingertip=project(K,middle_fingertip)

u_ring_first, v_ring_first=project(K,ring_first)
u_ring_second, v_ring_second=project(K,ring_second)
u_ring_third, v_ring_third=project(K,ring_third)
u_ring_finegrtip, v_ring_fingertip=project(K,ring_fingertip)

u_little_first, v_little_first=project(K,little_first)
u_little_second, v_little_second=project(K,little_second)
u_little_third, v_little_third=project(K,little_third)
u_little_finegrtip, v_little_fingertip=project(K,little_fingertip)

u_thumb_first,v_thumb_first=project(K,thumb_first)
u_thumb_second,v_thumb_second=project(K,thumb_second)
u_thumb_third,v_thumb_third=project(K,thumb_third)
u_thumb_finegrtip, v_thumb_fingertip=project(K,thumb_fingertip)

im=mpimg.imread('./data/dotted_hands/undistorted__image2.jpeg')
plt.imshow(im)
#plt.scatter(u_arm, v_arm, c='red', marker='.', s=20)
plt.scatter(u_index_first, v_index_first, c='red', marker='.', s=20)
plt.scatter(u_index_second, v_index_second, c='red', marker='.', s=20)
plt.scatter(u_index_third, v_index_third, c='red', marker='.', s=20)
plt.scatter(u_index_finegrtip, v_index_fingertip, c='red', marker='.', s=20)

plt.scatter(u_middle_first, v_middle_first, c='red', marker='.', s=20)
plt.scatter(u_middle_second, v_middle_second, c='red', marker='.', s=20)
plt.scatter(u_middle_third, v_middle_third, c='red', marker='.', s=20)
plt.scatter(u_middle_finegrtip, v_middle_fingertip, c='red', marker='.', s=20)

plt.scatter(u_ring_first, v_ring_first, c='red', marker='.', s=20)
plt.scatter(u_ring_second, v_ring_second, c='red', marker='.', s=20)
plt.scatter(u_ring_third, v_ring_third, c='red', marker='.', s=20)
plt.scatter(u_ring_finegrtip, v_ring_fingertip, c='red', marker='.', s=20)

plt.scatter(u_little_first, v_little_first, c='red', marker='.', s=20)
plt.scatter(u_little_second, v_little_second, c='red', marker='.', s=20)
plt.scatter(u_little_third, v_little_third, c='red', marker='.', s=20)
plt.scatter(u_little_finegrtip, v_little_fingertip, c='red', marker='.', s=20)

plt.scatter(u_thumb_first, v_thumb_first, c='red', marker='.', s=20)
plt.scatter(u_thumb_second, v_thumb_second, c='red', marker='.', s=20)
plt.scatter(u_thumb_third, v_thumb_third, c='red', marker='.', s=20)
plt.scatter(u_thumb_finegrtip, v_thumb_fingertip, c='red', marker='.', s=20)

plt.scatter(u, v, c='red', marker='.', s=20)
#plt.xlim(100,600)
#plt.ylim(600,300)
plt.show()

