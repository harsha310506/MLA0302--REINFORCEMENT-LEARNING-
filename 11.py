import cv2

img = cv2.imread(r"C:\Users\harsh\OneDrive\Pictures\WhatsApp Image 2026-02-08 at 10.01.08 PM (1).jpeg")

if img is None:
    print("Image not found!")
    exit()

rows, cols = img.shape[:2]

M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)

result = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow("Original", img)
cv2.imshow("Affine Transformation", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
