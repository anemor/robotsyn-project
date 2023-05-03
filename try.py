import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from common import *

transform_camera_base_point=rotate_z(np.deg2rad(-38))@translate_x(0,0,-52)
print(transform_camera_base_point)


