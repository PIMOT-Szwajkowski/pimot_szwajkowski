#!/usr/bin/env python

import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
import numpy as np

bridge = CvBridge()
refPt = []
colors = []

def filter_color(rgb_image, lower_bound_color, upper_bound_color):
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv", hsv_image)
    binary_image_mask = cv2.inRange(hsv_image, lower_bound_color, upper_bound_color)
    return binary_image_mask


def getContours(binary_image):
    _, contours, hierarchy = cv2.findContours(binary_image.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contour(rgb_image, contours):
    for c in contours:
        area = cv2.contourArea(c)
        if (area > 2000):
            cv2.drawContours(rgb_image, [c], -1, (255, 0, 255), 2)


def max_temp(max_temp_image):
    smallest = np.amin(max_temp_image)
    biggest = np.amax(max_temp_image)
    pixel = biggest
    return pixel


def on_mouse_click(event, x, y, flags, frame):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        refPt.append([x,y])
        colors.append(frame[y,x].tolist())
        print(colors)



def image_callback(ros_image):
    global bridge
    try:
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
        print(e)

    rgb_image = cv_image
    yellowLower = (30, 100, 50)
    yellowUpper = (60, 255, 255)
    binary_image_mask = filter_color(rgb_image, yellowLower, yellowUpper)
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)

    cv2.setMouseCallback('gray', on_mouse_click, gray)

    face_cascade = cv2.CascadeClassifier(
        '/home/nvidia/catkin_ws/src/pimot_szwajkowski/src/kamera/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        pole = w * h
        if (pole < 150000):
            contours = getContours(binary_image_mask)
            black_image = draw_contour(rgb_image, contours)
            for gray in faces:
                pixel = max_temp(gray)

            if pixel > 251:
                pixel = "> 36.6"
            else:
                pixel = "< 36.6"

            tekst = "Temperatura: " + str(pixel) + " stopni C"
            cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(rgb_image, tekst, (x, y - 10), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    mask = cv2.resize(rgb_image, (0, 0), fx=1, fy=1)
    cv2.imshow("Kamera termowizyjna", mask)
    cv2.waitKey(25)


def main(args):
    rospy.init_node('temperatura', anonymous=True)
    image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, image_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)

    # rosrun usb_cam usb_cam_node _video_device:=/dev/video3 _pixel_format:=yuyv