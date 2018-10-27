'''
@File CreditCardDetection.py
@Brief Detect shape of credit card in natural image
'''

import cv2 
import argparse
import numpy as np
from ContourDetect import Contour_Detector
from LineDetect import Corner_Detect_By_Hough_Line

def Save_Image(path, image):
	'''
	Save images by path

	# Parameters
	path: the path for saving images 
	image: image you want to save
	'''
	ok = cv2.imwrite(path, image)

	if ok:
		print(path, 'Saved!')
	else:
		print('Something wrong with', path)

def Canny_Detector(threshold):
	'''
	Canny detection by threshold values
	
	# Parameters
	threshold: take threshold for canny edge detection

	# Return
	edge_map: edge map generate from canny edge detection
	'''

	# Smooth image by gaussian filter
	img_blur = cv2.GaussianBlur(src, (5,5), 0)

	# Clean text by median filter
	img_blur = cv2.medianBlur(img_blur, 17)

	# Detect edges using canny detector
	edge_map = cv2.Canny(img_blur, threshold*0.66, threshold*1.33, apertureSize=3)

	cv2.imshow('Edge Map', edge_map)
	cv2.waitKey()
	cv2.destroyAllWindows()

	return edge_map


# Take arguments from command line 
parser = argparse.ArgumentParser()
parser.add_argument('-input', default='./images/test1.jpg')
args = parser.parse_args()

# Read input image
src = cv2.imread(args.input)
if src is None:
    print('Couldn\'t open', args.input)
    exit(0)

# Scale image by fixed height 500px 
ratio = 500/src.shape[0]
src = cv2.resize(src, (int(ratio*src.shape[1]), 500))

# Set threshold by image median value
threshold = np.mean(src)

# Get edge map by canny edge detection 
edge_map = Canny_Detector(threshold)

# Make contour complete by filling some small gaps on edge map
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
edge_map = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Edge Map', edge_map)
cv2.waitKey()
cv2.destroyAllWindows()
Save_Image('./images/output/test{}_edge_map.jpg'.format(args.input[-5]), edge_map)

# Get contour image on original image
img, contour, position = Contour_Detector(edge_map, src.copy())

# Capture bounding box contain credit card
x, y, w, h = position
bounding_box = contour[y-15:y+h+15,x-15:x+w+15]

cv2.imshow('Contour', bounding_box)
cv2.waitKey()
cv2.destroyAllWindows()
Save_Image('./images/output/test{}_contour.jpg'.format(args.input[-5]), contour)

# Hough Line Detection
corners = Corner_Detect_By_Hough_Line(bounding_box, position)

corner_plot = src.copy()
for p1, p2 in corners:
	cv2.circle(corner_plot, (p1, p2), 3, (0, 0, 255), -1)

cv2.imshow('Contour Plot', corner_plot)
cv2.waitKey()
cv2.destroyAllWindows()
Save_Image('./images/output/test{}_corner.jpg'.format(args.input[-5]), corner_plot)
