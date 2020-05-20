#!/usr/bin/env python

import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
import numpy as np


bridge = CvBridge()


def filter_color(rgb_image, lower_bound_color, upper_bound_color):
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    hsv = cv2.resize(hsv_image, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("2.HSV", hsv)
    mask = cv2.inRange(hsv_image, lower_bound_color, upper_bound_color)
    mask2 = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("3.Mask", mask2)
    return mask


def getContours(binary_image):
    _, contours, hierarchy = cv2.findContours(binary_image.copy(),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_ball_contour(binary_image, rgb_image, contours):
    black_image = np.zeros([binary_image.shape[0], binary_image.shape[1], 3], 'uint8')

    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if (area > 3000):
            cv2.drawContours(rgb_image, [c], -1, (150, 250, 150), 1)
            cv2.drawContours(black_image, [c], -1, (150, 250, 150), 1)
            cx, cy = get_contour_center(c)
            #cv2.rectangle(rgb_image, (cx, cy), (150,150), (0, 255, 0), 3)
            cv2.circle(rgb_image, (cx, cy), (int)(radius), (0, 0, 255), 1)
            cv2.circle(rgb_image, (cx, cy), 5, (150, 150, 255), -1)
            cv2.circle(black_image, (cx, cy), (int)(radius), (0, 0, 255), 1)
            cv2.circle(black_image, (cx, cy), 5, (150, 150, 255), -1)
    rgb_image2 = cv2.resize(rgb_image, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("4.RGB Image Contours", rgb_image2)
    black_image2 = cv2.resize(black_image, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("5.Black Image Contours", black_image2)


def get_contour_center(contour):
    M = cv2.moments(contour)
    cx = -1
    cy = -1
    if (M['m00'] != 0):
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    return cx, cy


def image_callback(ros_image):
    global bridge
    try:
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
        print(e)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(cv_image,'Dzien dobry',(50,50), font, 2,(0,255,255),5,cv2.LINE_AA)
    cv_image2 = cv2.resize(cv_image, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("1.CV", cv_image2)

    #yellowLower =(30, 100, 50)
    yellowLower =(30, 100, 10)
    yellowUpper = (60, 255, 255)
    binary_image_mask = filter_color(cv_image, yellowLower, yellowUpper)
    contours = getContours(binary_image_mask)
    draw_ball_contour(binary_image_mask, cv_image, contours)
    cv2.waitKey(25)


def main(args):
    rospy.init_node('szwajkowski', anonymous=True)
    image_sub = rospy.Subscriber("/zed/zed_node/left_raw/image_raw_color", Image, image_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)