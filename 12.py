import cv2
import numpy as np

img = cv2.imread(r"C:\Users\harsh\OneDrive\Pictures\virat -anushka .jpeg")

pts1 = np.float32([[50,50],[300,50],[50,300],[300,300]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, M, (300,300))

cv2.imshow("Original", img)
cv2.imshow("Perspective", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
