'''
@File FeatureExtract.py
@Brief Extract two features, corner and line, to find a more robust feature for detection
'''

import cv2
import numpy as np
import argparse

'''
Harris Corner Detection
'''

# Use Harris Corner Detection to detect corners
def CornerDetector(threshold):

	corners = cv2.cornerHarris(edge_map, 2, 11, 0.04)
	corners = cv2.dilate(corners, None)
	img = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)

	# Corner values over threshold will be red point
	img[corners>0.01*threshold*corners.max()] = [0,0,255]
	cv2.imshow(window_name, img)

'''
Hough Line Transform
'''

# Use Hough Line Transform to detect lines
def LineDetector(threshold):

	lines = cv2.HoughLines(edge_map, 1, (np.pi/180), 10*threshold)
	img = cv2.cvtColor(edge_map,cv2.COLOR_GRAY2RGB)
	
	# Draw red lines on image using theta and rho
	for rho,theta in lines[:,0,:]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
		cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)

	cv2.imshow(window_name, img)

if __name__=='__main__':

	# Take arguments from command line 
	parser = argparse.ArgumentParser()
	parser.add_argument('-input', default='./images/test3_edge_map.jpg')
	args = parser.parse_args()

	# Read input image
	edge_map = cv2.imread(args.input, cv2.CV_8UC1)
	if edge_map is None:
	    print('Couldn\'t open', args.input)
	    exit(0)

	# Edge map is a binary image
	_, edge_map = cv2.threshold(edge_map, 127, 255, cv2.THRESH_BINARY)

	# Parameter for CornerHarris tracker bar
	window_name = 'Corner Detector'
	title = 'Threshold'

	# Run Harris Corner Detector
	cv2.namedWindow(window_name)
	cv2.createTrackbar(title, window_name, 1, 100, CornerDetector)
	CornerDetector(1)
	cv2.waitKey()
	cv2.destroyAllWindows()

	# Parameter for LineDetector tracker bar
	window_name = 'Line Detector'
	title = 'Threshold'

	# Run Hough Line Transformation
	cv2.namedWindow(window_name)
	cv2.createTrackbar(title, window_name, 10, 20, LineDetector)
	LineDetector(10)
	cv2.waitKey()
	cv2.destroyAllWindows()
