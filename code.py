import cv2
import numpy as np
from skimage.filters import threshold_adaptive
from transform import correct_perspective_distortion
import helper
import argparse

# constructing the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--image", required = True, help = "Path to the image to be scanned")

# parsing the argument
args = vars(ap.parse_args())

# loading the image
image = cv2.imread(args["image"])

# making a copy of the image
orig = image.copy()

# computing the ratio of original image to new height
ratio = image.shape[0] / 650.0

# resizing the image
image = helper.resize(image, height = 650)

# convert the image to grayscale, blur it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
	
# added
# converting the image into black and white with 
# appropriate threshold value to remove illumination defect
ret,gray = cv2.threshold(gray,122,255,cv2.THRESH_BINARY)

# finding all the edges in the image
edged = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected image
print ("Detecting all the edges by canny edge detection.")
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.imshow("Edged", edged)
cv2.imwrite("edged.jpg",edged)
cv2.waitKey(0)	
cv2.destroyAllWindows()

#cv2.imshow("Gray",gray)
#cv2.imwrite("gray.jpg",gray)
#cv2.waitKey(0)

#Added
#ret,image = cv2.threshold(gray,114,255,cv2.THRESH_BINARY)

#cv2.imshow("Imf",image)
#cv2.waitKey(0)

# finding the contours in the edged image after canny edge detection
(_ ,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# sorting in descending order according to the length of the contour
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# looping over the contours
for c in cnts:
	#length of contour
	peri = cv2.arcLength(c, True)
	# approximate the contour
	approx = cv2.approxPolyDP(c, 0.05 * peri, True) # 0.05 is the extent to which i am approximating the contour
	
	# Added convex hull
	#approx = cv2.convexHull(c)

	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
	else:
		print ("No contour with four vertices found.")
		screenCnt = approx

# printing the 4 vertices of contour
print (screenCnt)

print ("Finding the required contour which covers the text.")
# drawing the contour on the image 
cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 2) # BGR =(0,0,255) = Red colour
cv2.imshow("Outline", image)
cv2.imwrite("contour.jpg",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the original image
print ("Correcting perspective distortion.")
output = correct_perspective_distortion(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
cv2.imshow("Originalshade.jpg",helper.resize(output,height=650))
cv2.waitKey(0)
cv2.imwrite("Orginalshade.jpg",output)
cv2.destroyAllWindows()
output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
output = threshold_adaptive(output, 251, offset = 10)
output = output.astype("uint8") * 255

# show the original and scanned images
cv2.imshow("Original", helper.resize(orig, height = 650))
cv2.imwrite("original.jpg",orig)
cv2.waitKey(0)
cv2.imshow("Scanned", helper.resize(output, height = 650))
cv2.imwrite("scanned.jpg",output)
cv2.waitKey(0)