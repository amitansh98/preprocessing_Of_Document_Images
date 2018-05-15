from pyimagesearch.transform import four_point_transform
from pyimagesearch import imutils
from skimage.filters import threshold_adaptive
import numpy as np
import argparse
import cv2

def brighten_image_hsv(image, global_mean_v):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(image_hsv)
    mean_v = int(np.mean(v))
    v = v - mean_v + global_mean_v
    image_hsv = cv2.merge((h, s, v))
    image_bright = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    return image_bright

image = cv2.imread("image/notes.jpg")
vs = []
for image_dir, image_name in get_next_image_loc(DATA_DIR):
    image = cv2.imread(os.path.join(DATA_DIR, image_dir, image_name))
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_hsv)
    vs.append(np.mean(v))
global_mean_v = int(np.mean(np.array(vs)))

brightened = brighten_image_hsv(resized, global_mean_v)
cv2.imshow("sc",brightened)