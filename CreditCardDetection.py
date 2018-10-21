'''
@File CreditCardDetection.py
@Brief Detect shape of credit card in natural image
'''

import cv2 
import argparse

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

	# Detect edges using canny detector
	edge_map = cv2.Canny(img_blur, threshold, threshold*3, 3)
	cv2.imshow(window_name, edge_map)


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

# Save the threshold we choose 
chosen_threshold = cv2.getTrackbarPos(title, window_name)
cv2.destroyAllWindows()


# Generate edge_map by chosen threshold
img_blur = cv2.GaussianBlur(src, (5, 5), 0)
edge_map = cv2.Canny(img_blur, chosen_threshold, chosen_threshold*3, 3)

# Save edge map image in './images' folder
cv2.imwrite("./images/test{}_edge_map.jpg".format(args.input[-5]), edge_map)
print("./images/test{}_edge_map.jpg saved!".format(args.input[-5]))