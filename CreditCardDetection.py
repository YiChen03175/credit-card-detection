'''
@File CreditCardDetection.py
@Brief Detect shape of credit card in natural image
'''

import cv2 
import argparse
import numpy as np
from ContourDetect import ContourDetector

def SaveImage(path, image):
	'''
	Save images by path

	# Parameter
	path: the path for saving images 
	image: image you want to save
	'''
	ok = cv2.imwrite(path, image)

	if ok:
		print(path, 'Saved!')
	else:
		print('Something wrong with', path)

def CannyDetector(threshold):
	'''
	Canny detection by threshold values
	
	# Parameter
	threshold: take threshold for canny edge detection

	# Return
	edge_map: edge map generate from canny edge detection
	'''
	img_blur=src
	# Smooth image by gaussian filter
	img_blur = cv2.GaussianBlur(img_blur,(5,5),0)

	# Clean text by median filter
	img_blur = cv2.medianBlur(img_blur, 17)

	# Detect edges using canny detector
	edge_map = cv2.Canny(img_blur, threshold*0.66, threshold*1.33, apertureSize=3)

	cv2.imshow('Edge Map', edge_map)

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

# Set threshold by gray scale image median value
threshold = np.median(src)

# Get edge map by canny edge detection 
edge_map = CannyDetector(threshold)

# Make contour complete by filling some small gaps on edge map
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
edge_map = cv2.dilate(edge_map, kernel)
edge_map = cv2.erode(edge_map, kernel)

cv2.imshow('Edge Map', edge_map)
cv2.waitKey()
cv2.destroyAllWindows()
SaveImage('./images/output/test{}_dilate.jpg'.format(args.input[-5]), edge_map)

# Get contour image on original image
img, contour = ContourDetector(edge_map, src)

cv2.imshow('Contour', img)
cv2.waitKey()
cv2.destroyAllWindows()
SaveImage('./images/output/test{}_contour.jpg'.format(args.input[-5]), img)
