#
# The code in this script is largely copied from the official tutorial
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
#
# Please read the tutorial for explanations of the OpenCV functions.

import numpy as np
import cv2 as cv
import glob
from os.path import join, basename, realpath, dirname, exists, splitext
 
# captures video frame by frame https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/
def extract_image_from_video(video_path, out_folder):
    video = cv.VideoCapture(video_path)
    success, image = video.read()
    count = 0
    print('Extracting images from video\n')
    while(success):
        video.set(cv.CAP_PROP_POS_MSEC, (count*1000))       # to capture a frame every second
        cv.imwrite(out_folder + 'image%d.jpg' % count, image)
        success, image = video.read()
        print('     Read frame #', count, ':', success)
        count += 1
    video.release()

def undistort_images(results, images_folder, out_folder):
    ok, K, dc, rvecs, tvecs, std_int, std_ext, per_view_errors = results

    image_paths = glob.glob(images_folder)
    for image_path in sorted(image_paths):
        print("%s..." % basename(image_path), end='')
        test = 'data/hands/image0.jpg'
        I = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        cv.imshow('image', I)

        dst = cv.undistort(I, K, dc)
        cv.imwrite(out_folder + 'undistorted__' + basename(image_path), dst)

def detect_checkerboard(board_size, square_size, calibrate_flags, detect_flags, subpix_criteria, image_path, out_folder):
    # Detect checkerboard points
    # Note: This first tries to use existing checkerboard detections,
    if exists(join(out_folder, 'u_all.npy')):
        u_all = np.load(join(out_folder, 'u_all.npy'))
        X_all = np.load(join(out_folder, 'X_all.npy'))
        image_size = np.loadtxt(join(out_folder, 'image_size.txt')).astype(np.int32)
        print('Using previous checkerboard detection results.')
    else:
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ... (6,5,0)
        X_board = np.zeros((board_size[0]*board_size[1], 3), np.float32)
        X_board[:,:2] = square_size*np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        X_all = []
        u_all = []
        image_size = None
        image_paths = glob.glob(image_path)
        for image_path in sorted(image_paths):
            print('%s...' % basename(image_path), end='')

            I = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            if not image_size:
                image_size = I.shape
            elif I.shape != image_size:
                print('Image size is not identical for all images.')
                print('Check image "%s" against the other images.' % basename(image_path))
                quit()
            ok, u = cv.findChessboardCorners(I, (board_size[0],board_size[1]), detect_flags)
            if ok:
                print('detected all %d checkerboard corners.' % len(u))
                X_all.append(X_board)
                u = cv.cornerSubPix(I, u, (11,11), (-1,-1), subpix_criteria)
                u_all.append(u)
            else:
                print('failed to detect checkerboard corners, skipping.')

        np.savetxt(join(out_folder, 'image_size.txt'), image_size)
        #np.savetxt(join('/data/calibration', 'image_size.txt'), image_size)

        np.save(join(out_folder, 'u_all.npy'), u_all) # Detected checkerboard corner locations
        np.save(join(out_folder, 'X_all.npy'), X_all) # Corresponding 3D pattern coordinates
    return X_all, u_all, image_size


def find_mean_errors(results, out_folder):
    ok, K, dc, rvecs, tvecs, std_int, std_ext, per_view_errors = results

    mean_errors = []
    for i in range(len(X_all)):
        u_hat, _ = cv.projectPoints(X_all[i], rvecs[i], tvecs[i], K, dc)
        vector_errors = (u_hat - u_all[i])[:,0,:] # the indexing here is because OpenCV likes to add extra dimensions.
        scalar_errors = np.linalg.norm(vector_errors, axis=1)
        mean_errors.append(np.mean(scalar_errors))

    np.savetxt(join(out_folder, 'K.txt'), K) # Intrinsic matrix (3x3)
    np.savetxt(join(out_folder, 'dc.txt'), dc) # Distortion coefficients
    np.savetxt(join(out_folder, 'mean_errors.txt'), mean_errors)
    np.savetxt(join(out_folder, 'std_int.txt'), std_int) # Standard deviations of intrinsics (entries in K and distortion coefficients)
    print('Calibration data is saved in the folder "%s"' % realpath(out_folder))

#======================= MAIN SCRIPT  ========================

# extract_image_from_video(video_path='data/hands_video/IMG_0655.mov', out_folder='data/hands_video/')
# print("FERDIG MED VIDEO\n")
#------------------------ EXTRACT IMAGES ------------------------
video_path = 'data/calibration/calibration_vid.mov'
out_images_path = 'data/calibration/'
#extract_image_from_video(video_path, out_images_path)

#------------------------ CALIBRATE CAMERA ------------------------
image_path_pattern = 'data/calibration/*.jpg'
output_folder = dirname(image_path_pattern)

# CALIBRATE = True
# results = None
board_size = np.array([9,6]) # internal corners
square_size = 1 # mm. Real world length of the sides of the squares

calibrate_flags = 0 # Use default settings (three radial and two tangential)
# calibrate_flags = cv.CALIB_ZERO_TANGENT_DIST|cv.CALIB_FIX_K3 # Disable tangential distortion and third radial distortion coefficient

detect_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK # Flags to findChessboardCorners to improve performance
subpix_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Termination criteria for cornerSubPix routine

X_all, u_all, image_size = detect_checkerboard(board_size, square_size, calibrate_flags, detect_flags, subpix_criteria, image_path_pattern, output_folder)

print('Calibrating. This may take a minute or two...', end='')
results = cv.calibrateCameraExtended(X_all, u_all, image_size, None, None, flags=calibrate_flags)
print('Done!')

ok, K, dc, rvecs, tvecs, std_int, std_ext, per_view_errors = results

#print('Calibration results are saved in the folder "%s"' % realpath(output_folder))
#results = 
#------------------------ CALIBRATION ERRROS ------------------------
find_mean_errors(results, output_folder)

#------------------------ UNDISTORT IMAGES USING CALIBRATION ------------------------
print("Undistort\n")
# undistort_images(results, images_folder='data/hands/*.jpg', out_folder='data/undistorted/')
undistort_images(results, images_folder='data/hands_video/*.jpg', out_folder='data/hands_video/')