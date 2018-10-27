'''
@File ContourDetector.py
@Brief Detec the shape of credit card
'''

import cv2
import argparse
import numpy as np

def Contour_Detector(edge_map, background):
	'''
	Detect largest contour on edge map

	# Parameters
	edge_map: edge map by canny detection

	# Return
	img: edge_map with green approximate contour
	approx: approximation of contour provided by findContours
	pos: a rectangle box bounding the main object, contain info (x coordinate, y coordinate, width, height)
	'''

	# Find the largest area contour on edge map
	_, contours, _ = cv2.findContours(edge_map.copy(), cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
	
	# Restrict the approximation by contour's length
	length = cv2.arcLength(contours, True)
	approx = cv2.approxPolyDP(contours, 0.004*length, True)

	# Take the parameters to narrow down the contour
	pos = cv2.boundingRect(contours)

	# Draw contour on background image
	if len(background.shape) == 2:
		background = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
	cv2.drawContours(background, [approx], -1, (0, 255, 0), 2)
	
	# Draw contour on a blank image (black)
	blank = np.zeros(edge_map.shape, np.uint8)
	cv2.drawContours(blank, [approx], -1, 255, 2)

	cv2.imshow('Contour', background)
	cv2.waitKey()
	cv2.destroyAllWindows()

	return background, blank, pos