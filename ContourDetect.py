'''
@File ContourDetector.py
@Brief Detec the shape of credit card
'''

import cv2
import argparse
import numpy as np
from itertools import combinations

def Contour_Detector(edge_map):
	'''
	Detect largest contour on edge map

	# Parameters
	edge_map: edge map by canny detection

	# Return
	MIN_CNT: the smallest contour satisfied all ceritions.
	pos: a rectangle box bounding the main object, contain info (x coordinate, y coordinate, width, height)
	'''

	# Find the largest area contour on edge map
	_, contours, _ = cv2.findContours(edge_map.copy(), cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
	contours= sorted(contours, key=cv2.contourArea, reverse=True)[:3]
	contours = np.array(contours)

	# Combination of contours , generate index like [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
	combs = []
	for c in range(3):
		for comb in list(combinations(range(3), c+1)):
			combs.append(list(comb))

	# Initialize the minimum area size and corresponding contour
	MIN, MIN_CNT = 1e10, None
	test = []
	for comb in combs: 

		# Generate contour from combination
		cnt = np.concatenate((contours[comb]), axis=0)

		# Generate their convex contour
		hull = cv2.convexHull(cnt)

		# Approximation of convex contour
		length = cv2.arcLength(hull, True)
		approx = cv2.approxPolyDP(hull, 0.02*length, True)

		# Calculate convex contour area size
		hull_size = cv2.contourArea(hull)

		# Calculate the percentage of contour area in whole image
		hull_percentage = cv2.contourArea(hull)/(edge_map.shape[0]*edge_map.shape[1])

		# Criterion for selecting proper contour
		if hull_percentage>0.05 and len(approx)==4:
			if MIN_CNT is None:
				MIN = hull_size
				MIN_CNT = hull
			else:
				# Avoid contour shrink and allow slight expansion
				area_sub = (MIN-hull_size)/hull_size
				if (hull_size<MIN and area_sub>0.05) or -area_sub<0.05:
					MIN = hull_size
					MIN_CNT = hull

	# if there are no contour satisfied, return None
	if MIN==1e10: return None, None

	# Bounding box info of contour
	pos = cv2.boundingRect(MIN_CNT)

	return MIN_CNT, pos