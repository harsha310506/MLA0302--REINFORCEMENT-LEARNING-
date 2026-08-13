import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    h, w = frame.shape[:2]

    pts1 = np.float32([[50,50],[w-50,50],[50,h-50],[w-50,h-50]])
    pts2 = np.float32([[0,0],[400,0],[0,400],[400,400]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, M, (400,400))

    cv2.imshow("Video", frame)
    cv2.imshow("Perspective", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
