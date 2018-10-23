'''
@File ContourDetector.py
@Brief Detec the shape of credit card
'''

import cv2
import argparse
import numpy as np

def ContourDetector(edge_map, background):
	'''
	Detect largest contour on edge map

	# Parameter
	edge_map: edge map by canny detection

	# Return
	img: edge_map with green approximate contour
	approx: approximation of contour provided by findContours
	'''

	# Find the largest area contour on edge map
	_, contours, h_ = cv2.findContours(edge_map.copy(), cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key = cv2.contourArea, reverse = True)[0]

	# Restrict the approximation by contour's length
	length = cv2.arcLength(contours, True)
	approx = cv2.approxPolyDP(contours, 0.005*length, True)

	# Draw contour on background image
	if len(background.shape) == 2:
		background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
	cv2.drawContours(background, [approx], -1, (0, 255, 0), 2)
	
	cv2.imshow('Contour', background)

	return background, approx

if __name__=='__main__':

	# Take arguments from command line 
	parser = argparse.ArgumentParser()
	parser.add_argument('-input', default='./images/test1_edge_map.jpg')
	args = parser.parse_args()

	# Read input image
	edge_map = cv2.imread(args.input)
	edge_map = cv2.cvtColor(edge_map,cv2.COLOR_RGB2GRAY)

	# Edge map is a binary image
	_,edge_map = cv2.threshold(edge_map, 127, 255, cv2.THRESH_BINARY)

	image = ContourDetector(edge_map, edge_map)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
