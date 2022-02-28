from matplotlib.pyplot import sci

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


def detector(img, classifier, scaleFactor=None, minNeighbors=None):
    result = img.copy()
    rects = classifier.detectMultiScale(
        result, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    for (x, y, w, h) in rects:
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 255, 255))

    return result


conf = cv2.imread("./data/solvay_conference.jpg")
cooper = cv2.imread("./data/cooper.jpg")
car = cv2.imread("./data/car_plate.jpg")

face_cascade = "./data/object_detection/haarcascades/haarcascade_frontalface_default.xml"
eye_cascade = "./data/object_detection/haarcascades/haarcascade_eye.xml"
lbp_cascade = "./data/object_detection/lbpcascades/lbpcascade_frontalface.xml"
car_cascade = "./data/object_detection/haarcascades/haarcascade_russian_plate_number.xml"

face_classifier = cv2.CascadeClassifier(face_cascade)
eye_classifier = cv2.CascadeClassifier(eye_cascade)
car_classifier = cv2.CascadeClassifier(car_cascade)

cam = cv2.VideoCapture(1)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera_eye", cv2.WINDOW_NORMAL)

# def pixel_face(img, b=3):
#     (height, widht) = img.shape[:2]
#     x = np.linspace(0, widht, b + 1, dtype="int")
#     y = np.linspace(0, height, b + 1, dtype="int")
#     # for i in range(1, len(y)):
#     #     for j in range(1, len(x)):
#     #         X_1 = x[j - 1]
#     #         Y_1 = y[i - 1]
#     #         X_2 = x[j]
#     #         Y_2 = y[i]

#     #         ROI = img[Y_1:Y_2, X_1:X_2]
#     #         (B, G, R) = [int(k) for k in cv2.mean(ROI)[:3]]
#     #         cv2.rectangle(img, (X_1, Y_1), (X_2, Y_2),
#     #             (B, G, R), -1)
#     return img

while cam.isOpened():
    ret, frame = cam.read()
    face_detector = detector(frame, face_classifier, 1.2, 5)
    eye_detector = detector(frame, eye_classifier, 1.2, 5)


    # result = detector(frame, eye_classifier, 1.2, 5)
    # print(result)
    cv2.imshow("Camera", face_detector)
    cv2.imshow("Camera_eye", eye_detector)


    for x, y, w, h in eye_detector:
        pass

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

# result = detector(car, car_classifier, 1.2, 5)

# plt.figure()
# plt.imshow(result)
# plt.show()
