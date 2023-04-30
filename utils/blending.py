import cv2
import numpy as np

img1 = cv2.imread('..\\asset\\0.jpg')
img2 = cv2.imread('..\\asset\\0_OUT.jpg')
dst = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

cv2.imwrite("..\\asset\\blended_image.jpg",dst)
cv2.imshow('Blended_image',dst)

cv2.waitKey(0)
cv2.destroyAllWindows()