import cv2
import time
import numpy as np

cam = cv2.VideoCapture(1)
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Binary", cv2.WINDOW_KEEPRATIO)

bounds = {
    "red": {"lower": (0, 150, 140), "upper": (15, 255, 255)},
    "blue": {"lower": (90, 160, 120), "upper": (130, 255, 255)},
    "green": {"lower": (65, 190, 80), "upper": (120, 255, 255)},
    "yellow": {"lower": (20, 80, 160), "upper": (35, 255, 255)}
}

def find_markers():
    mask = cv2.inRange(hsv, bounds["blue"]['lower'], bounds["blue"]['upper'])
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    reg, fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
    fg = np.uint8(fg)
    confuse = cv2.subtract(mask, fg)
    ret, markers = cv2.connectedComponents(fg)
    markers += 1
    markers[confuse == 255] = 0

    return markers

while cam.isOpened():
    _, image = cam.read()
    curr_time = time.time()
    blurred = cv2.GaussianBlur(image, (11, 11), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    markers = find_markers()

    wmarkers = cv2.watershed(image, markers.copy())
    contours, hierarchy = cv2.findContours(
        wmarkers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(image, contours, i, (0, 255, 0), 6)

            print(int((len(hierarchy[0])-1) / 2))

    cv2.imshow("Camera", image)
    cv2.imshow("Binary", np.float32(wmarkers))

    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
