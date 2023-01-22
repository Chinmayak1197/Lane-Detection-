import cv2 
import numpy as np 
import matplotlib.pyplot as plt

def canny(img):

	lane_image = np.copy(img)
	gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)

	# Reduce Noise : Gaussian Blur 
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	# Canny edge : internal (5,5) gaussian is applied when cannyedge is used. 
	# computes the derivaties to find the change in pixel 

	low_threshold = 50
	high_threshold = 150
	canny = cv2.Canny(img,low_threshold,high_threshold)

	return canny

def region_of_interest(image):
	
	height = image.shape[0]
	width = image.shape[1]
	triangle = np.array([[(200,height), (1100, height), (550,250)]])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, triangle, 255) 
	masked = cv2.bitwise_and(image, mask)
	return masked

def display_lines(image, lines):
	lane_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			x1,y1,x2,y2 = line.reshape(4)
			cv2.line(lane_image,(x1,y1),(x2,y2), (255,0,0), 10)
	return lane_image


image = "/home/chinmay/Desktop/finding_lanes/test_image.jpg"

img = cv2.imread(image)
lane_image = np.copy(img)

# Canny Edge Detection : Identify sharp changes in the intensity in adjacent pixels 
# Gradient : Measure of change in the brightness over adjacent pixels 
# Strong gradient : 0 -> 255
# Small gradient : 0 -> 15 

# 1. Convert to grayscale, has only one channel, each pixel will have onky one intensity value. Processing a single channel is faster than 3 channel. 

canny = canny(img)


# Hough Transforms : Identify the lanes lines 
# Reducing to the region of interest : block a mask to just take the region which we are interest
# Bitwise and is use just to keep mask from the image 

# Hough Transofrms : y = mx+b -> y,x -> b,m
# Change of x cannot be encounter when its 0 as slope will be infinite so we change to polar coordinates.
cropped = region_of_interest(canny)
lines = cv2.HoughLinesP(cropped, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
line_image = display_lines(lane_image, lines)

# Blend to the orignal color image 
combine_image = cv2.addWeighted(lane_image, 0.8, line_image, 1,1)


# plt.imshow(canny)
# plt.show()
cv2.imshow("result", combine_image)
cv2.waitKey(0) #Display Infinitly 