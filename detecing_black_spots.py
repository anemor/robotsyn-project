import cv2
import numpy as np
import matplotlib.pyplot as plt

img_rgb = cv2.imread("./data/hand_cropped.jpg")
print(np.shape(img_rgb))
#plt.imshow(img_rgb)
#plt.show()
#cv2.imshow('img',imS)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

'''img_gray_norm=img_gray/255
threshold = 0.3
#w, h = img_gray.shape[::-1]
#loc = np.where( img_gray >= threshold)
points=img_gray_norm<threshold
for pty in range(len(points)):
    for ptx in range(len(points[pty])):
        if(points[pty][ptx]):
            img_rgb[pty][ptx]=[0,255,255]

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)'''

gray_blurred = cv2.blur(img_gray, (3, 3))
detected_circles = cv2.HoughCircles(gray_blurred, 
                   cv2.HOUGH_GRADIENT, 1, 100, param1 = 200,
               param2 = 11, minRadius = 0, maxRadius = 20)
if detected_circles is not None:
  
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))
  
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
  
        # Draw the circumference of the circle.
        cv2.circle(img_rgb, (a, b), r, (0, 255, 0), 2)
  
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(img_rgb, (a, b), 1, (0, 0, 255), 3)
        imS = cv2.resize(img_rgb, (636, 583))
        cv2.imshow("Detected Circle", imS)
        cv2.waitKey(0)

img_gray_norm=img_gray/255
threshold = 0.3
#w, h = img_gray.shape[::-1]
#loc = np.where( img_gray >= threshold)
points=img_gray_norm<threshold
for pty in range(len(points)):
    for ptx in range(len(points[pty])):
        if(points[pty][ptx]):
            img_rgb[pty][ptx]=[0,255,255]
#for pt in zip(*loc[::-1]):
#    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 0)
#plt.imshow(img_rgb)
#plt.show()
imS = cv2.resize(img_rgb, (636, 583))                # Resize image
cv2.imshow('Detected',imS)

#cv2.imwrite("D:/4/Detect/"+str(i)+".0.jpg",img_rgb)
#th2 = cv2.adaptiveThreshold(img_rgb,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#            cv2.THRESH_BINARY,11,2)
#th3 = cv2.adaptiveThreshold(img_rgb,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY,11,2)
#images = [th2, th3]
#inv = ~th3
#res = cv2.bitwise_and(img_rgb,inv)
#cv2.imshow(res)
cv2.waitKey(0)
#cv2.imwrite("result.jpg",res)
cv2.destroyAllWindows()

'''# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (3, 3))
  
# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred, 
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
               param2 = 30, minRadius = 1, maxRadius = 40)
  
# Draw circles that are detected.
if detected_circles is not None:
  
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))
  
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
  
        # Draw the circumference of the circle.
        cv2.circle(img, (a, b), r, (0, 255, 0), 2)
  
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
        cv2.imshow("Detected Circle", img)
        cv2.waitKey(0)'''

'''img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('a.jpg',0)
w, h = template.shape[::-1]
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.****
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 0)
cv2.imshow('Detected',img_rgb)
cv2.imwrite("D:/4/Detect/"+str(i)+".1.jpg",img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

'''import cv2 as cv
img = cv.imread('med.jpg',0)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
titles = ['Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [th2, th3]
inv = ~th3
res = cv.bitwise_and(img,inv)
cv.imshow(titles[1],res)
cv.waitKey(0)
cv.imwrite("result.jpg",res)
cv.destroyAllWindows()'''