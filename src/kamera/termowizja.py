#!/usr/bin/env python

import numpy as np
import cv2
import time

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys

def filter_color(rgb_image, lower_bound_color, upper_bound_color):
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv image", hsv_image)
    mask2 = cv2.inRange(hsv_image, lower_bound_color, upper_bound_color)
    return mask2

def getContours(binary_image):
    _, contours, hierarchy = cv2.findContours(binary_image.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contour(binary_image, mask, contours):
    black_image = np.zeros([binary_image.shape[0], binary_image.shape[1], 3], 'uint8')
    for c in contours:
        area = cv2.contourArea(c)
        if (area > 3000):
            cv2.drawContours(black_image, [c], -1, (255, 0, 255), 2)
            cv2.drawContours(mask, [c], -1, (255, 0, 255), 2)
    return black_image

def detect_temperature_face(image_frame):

    yellowLower = (30, 100, 50)
    yellowUpper = (60, 255, 255)
    rgb_image = image_frame
    binary_image_mask = filter_color(rgb_image, yellowLower, yellowUpper)
    contours = getContours(binary_image_mask)

    mask = cv2.resize(rgb_image, (0, 0), fx=1, fy=1)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('/home/nvidia/catkin_ws/src/pimot_szwajkowski/src/kamera/haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(mask,'Temperatura: ',(x,y-10), font, 0.7,(0,255,255),2,cv2.LINE_AA)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

    black_image = draw_contour(binary_image_mask, mask, contours)

    for (x, y, w, h) in faces:
        cv2.rectangle(black_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Kamera termowizyjna", mask)
    #cv2.imshow("Black image", black_image)

def main():

    video_capture = cv2.VideoCapture('/home/nvidia/catkin_ws/src/pimot_szwajkowski/src/kamera/movie/patryk.mov')

    while (True):
        ret, frame = video_capture.read()
        detect_temperature_face(frame)
        time.sleep(0.033)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

cv2.waitKey(0)
cv2.destroyAllWindows()