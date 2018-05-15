import cv2
import numpy as np

def points_in_cyclic_order(pts):

	# Initialize a new array of size 4 (vertices) and each vertex has x and y coordinates. 
	rect = np.zeros((4, 2), dtype = "float32")

	# Sum the x and y coordinates of each vertex
	s = pts.sum(axis = 1)

	# the top-left point will have the smallest sum
	rect[0] = pts[np.argmin(s)]  # np.argmin returns the index of the minimum value along an axis.
	
	# the bottom-right point will have the largest sum
	rect[2] = pts[np.argmax(s)]

	# computing the difference between the points
	diff = np.diff(pts, axis = 1)

	# top-right point will have the smallest difference

	rect[1] = pts[np.argmin(diff)]

	# whereas the bottom-left will have the largest difference
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def correct_perspective_distortion(image, points):
	# obtaining the vertices in cyclic order
	vertices = points_in_cyclic_order(points)
	(tl, tr, br, bl) = vertices

	# calculating the width of the output image
	width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(width1), int(width2))

	# calculating the height of the output image
	height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(height1), int(height2))

	# assigning the vertices of our output image to an array
	dst = np.array([
		[0, 0],
		[maxWidth , 0],
		[maxWidth , maxHeight ],
		[0, maxHeight ]], dtype = "float32")

	# computing the perspective transformation matrix
	M = cv2.getPerspectiveTransform(vertices, dst)

	# applying the transformation matrix on input image
	corrected = cv2.warpPerspective(image, M, (maxWidth, maxHeight)) #image,transformation matrix,dimensions

	return corrected