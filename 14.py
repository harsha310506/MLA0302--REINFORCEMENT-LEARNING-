import cv2
import numpy as np

img = cv2.imread(r"C:\Users\harsh\OneDrive\Pictures\porsche.jpg")

pts1 = np.float32([[50,50],[300,50],[50,300],[300,300]])
pts2 = np.float32([[30,70],[280,40],[70,320],[320,300]])

H, _ = cv2.findHomography(pts1, pts2)

result = cv2.warpPerspective(img, H, (400,400))

cv2.imshow("Original", img)
cv2.imshow("Homography", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
