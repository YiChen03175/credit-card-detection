'''
@File LineDetection.py
@Brief Detect four corners of object by hough transform
'''

import cv2
import argparse
import numpy as np

def Draw_lines(img, rho, theta, color=(0,0,255),const=1000):
	'''
	Draw a line on images by rho and theta

	# Parameters
	img: image be drawn
	rho, theta: parameter to draw a line
	color: line color
	const: constant according to image size

	# Return
	img: image with drawn line
	'''
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + const*(-b)) 
	y1 = int(y0 + const*(a))
	x2 = int(x0 - const*(-b))
	y2 = int(y0 - const*(a))
	cv2.line(img, (x1, y1), (x2, y2), color, 2)

	return img

def intersection(line1, line2):
	'''
	Finds the intersection of two lines by rho and theta

	# Parameters
	line1, line2: contain rho and theta, which represent this two lines

	# Return
	x0, y0: the intersection coordinate
	'''
	rho1, theta1 = line1
	rho2, theta2 = line2
	A = np.array([
	    [np.cos(theta1), np.sin(theta1)],
	    [np.cos(theta2), np.sin(theta2)]
	])
	b = np.array([[rho1], [rho2]])
	x0, y0 = np.linalg.solve(A, b)
	x0, y0 = int(np.round(x0)), int(np.round(y0))
	return x0, y0

def Corner_Detect_By_Hough_Line(img, pos):
	'''
	Find corners by using hough line detection

	# Parameters
	img: binary image contain only object contour
	pos: object position on image, contain a rectangle info (x coordinate, y coordinate, width, height)

	# Return
	centers: object four corners position on original image, shape=(4,2) 
	'''
	
	# Control hough line detection threshold according to their image size
	if pos[2]<150 or pos[3]<150:
		line_thresh, num = 80, 0
		# Should generate at least 30 lines by this threshold
		while num<30:
			lines = cv2.HoughLines(img, 1, (np.pi/180), line_thresh)
			line_thresh -= 3
			if lines is None:
				num = 0
			else:
				num = lines.shape[0]
	else:
		line_thresh, num = 80, 100
		# Should generate at most 15 lines by this threshold
		while num>15:
			lines = cv2.HoughLines(img, 1, (np.pi/180), line_thresh)
			line_thresh += 5
			if lines is None:
				num = 0
			else:
				num = lines.shape[0]

	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)	

	# Take all theta from lines
	angles = np.array([line[0][:] for line in lines])

	# Multiply angle by two because degree (1, 359) are actually closed
	angles = np.array([[np.cos(angle*2), np.sin(angle*2)] for rho, angle in angles], dtype=np.float32)

	# K-means for classify vertical and horizontal lines
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	flags = cv2.KMEANS_RANDOM_CENTERS
	labels,centers = cv2.kmeans(angles, 2, None, criteria, 10, flags)[1:]
	
	# Seperate vertical and horizontal lines
	line_group = {}
	for label in range(2):
		line_group[label] = lines[labels==label]

	# Find all intersection from 
	points = []
	for line1 in line_group[0]:
		for line2 in line_group[1]:
			point = intersection(line1, line2)
			points.append(point)

	points = np.array(points, dtype=np.float32)
	labels,centers = cv2.kmeans(points, 4, None, criteria, 10, flags)[1:]
	centers = np.int0(np.round(centers))

	# Draw corners on image
	corner_plot = img.copy()
	for p1, p2 in centers:
		cv2.circle(corner_plot, (p1, p2), 3, (0, 0, 255), -1)

	cv2.imshow('corner', corner_plot)
	cv2.waitKey()
	cv2.destroyAllWindows()
	cv2.imwrite('./images/output/corner_plot.jpg', corner_plot)


	# Correct right position on original image
	centers += [pos[0]-15, pos[1]-15]

	return centers



if __name__=='__main__':

	# Take arguments from command line 
	parser = argparse.ArgumentParser()
	parser.add_argument('-input', default='./images/output/contour.jpg')
	args = parser.parse_args()

	# Read input image
	edge_map = cv2.imread(args.input, cv2.CV_8UC1)
	ret,thresh = cv2.threshold(edge_map, 127, 255, cv2.THRESH_BINARY)

	# Find four corners by hough transform
	corners = Corner_Detect_By_Hough_Line(thresh, [0, 0 ,edge_map.shape[0], edge_map.shape[1]])

	print(corners)