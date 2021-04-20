from types import DynamicClassAttribute
import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

k = 8

while True:
    _, frame = cap.read()

    Z = frame.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(
        Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))

    cv2.imshow('res2', res2)

    # press s for save image
    if cv2.waitKey(1) == ord("s"):
        now = time.time()
        name = "image{}.jpg".format(now)
        cv2.imwrite(name, res2)

    # press q for exit
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
