'''
@File CreditCardDetection.py
@Brief Detect shape of credit card in natural image
'''

import cv2 
import argparse
import numpy as np
from ContourDetect import ContourDetector

def CannyDetector(threshold=40):
	'''
	Canny detection by threshold values
	
	# Parameter
	threshold: minimum threshold for canny detector, maximum threshold is multiply by 3
	'''

	# Smooth image by gaussian filter
	img_blur = cv2.GaussianBlur(src, (5, 5), 0)

	# Clean text by median filter
	img_blur = cv2.medianBlur(img_blur, 23)

	# Detect edges using canny detector
	edge_map = cv2.Canny(img_blur, threshold, threshold*3, apertureSize = 3)

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

'''
# Parameters for canny dectector tracker bar
window_name = 'Edge Map'
title = 'Threshold:'

# Create window for trackerbar
print('Choose a proper threshold and press any key ...')
cv2.namedWindow(window_name)
cv2.createTrackbar(title, window_name , 1, 100, CannyDetector)
CannyDetector(1)
cv2.waitKey()

# Save the threshold we choose before destroy
chosen_threshold = cv2.getTrackbarPos(title, window_name)
cv2.destroyAllWindows()
'''

# Generate edge_map by chosen threshold
edge_map = CannyDetector()

# Make outline more obvious 
edge_map = cv2.dilate(edge_map, None)

cv2.imshow('Edge Map', edge_map)
cv2.waitKey()
cv2.destroyAllWindows()

# Save edge map image in './images' folder
cv2.imwrite('./images/test{}_edge_map.jpg'.format(args.input[-5]), edge_map)
print('./images/test{}_edge_map.jpg saved!'.format(args.input[-5]))

# Get contour image on original image
img, contour = ContourDetector(edge_map, src)

cv2.imshow('Contour', img)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('./images/test{}_contour.jpg'.format(args.input[-5]), img)
print('./images/test{}_contour.jpg saved!'.format(args.input[-5]))