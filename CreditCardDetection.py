'''
@File CreditCardDetection.py
@Brief Detect shape of credit card in natural image
'''

import cv2 
import argparse
import numpy as np

'''
Canny Edge Detection
'''

# Parameters for canny dectector tracker bar
window_name = 'Edge Map'
title = 'Threshold:'

# Show canny detection with different low threshold values
def CannyDetector(threshold):

	# Smooth image by gaussian filter
	img_blur = cv2.GaussianBlur(src, (5, 5), 0)

	# Clean text by median filter
	img_blur = cv2.medianBlur(img_blur, 17)

	# Detect edges using canny detector
	edge_map = cv2.Canny(img_blur, threshold, threshold*3, 3)
	cv2.imshow(window_name, edge_map)

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


# Create window for trackerbar
print('Choose a proper threshold and press any key ...')
cv2.namedWindow(window_name)
cv2.createTrackbar(title, window_name , 1, 100, CannyDetector)
CannyDetector(1)
cv2.waitKey()

# Save the threshold we choose before destroy
chosen_threshold = cv2.getTrackbarPos(title, window_name)
cv2.destroyAllWindows()

# Generate edge_map by chosen threshold
edge_map = CannyDetector(chosen_threshold)

# Save edge map image in './images' folder
cv2.imwrite('./images/test{}_edge_map.jpg'.format(args.input[-5]), edge_map)
print('./images/test{}_edge_map.jpg saved!'.format(args.input[-5]))

'''
Hough Line Transform
'''

lines = cv2.HoughLines(edge_map, 1, (np.pi/180), 100)
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
cv2.waitKey()

# Save hough line image 
cv2.imwrite('./images/test{}_hough_line.jpg'.format(args.input[-5]), img)
print('./images/test{}_hough_line.jpg saved!'.format(args.input[-5]))