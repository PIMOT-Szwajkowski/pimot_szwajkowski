#!/usr/bin/env python

import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
import numpy as np

bridge = CvBridge()

def image_callback(ros_image):
    global bridge
    try:
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
        print(e)
    mask = cv2.resize(cv_image, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Obraz5", gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0) #w nawiasie macierz K min (1,1), nieparzyste
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 1000:
            continue
        cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(mask, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.drawContours(mask, contours, -2, (0, 222, 0), 1)

    cv2.imshow("Obraz", mask)
    cv2.imshow("Obraz2", dilated)
    cv2.waitKey(25)

def main(args):
    rospy.init_node('face_detection_mv', anonymous=True)
    image_sub = rospy.Subscriber("/zed/zed_node/left_raw/image_raw_color", Image, image_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)