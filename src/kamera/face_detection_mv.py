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
    face_cascade = cv2.CascadeClassifier('/home/nvidia/catkin_ws/src/pimot_szwajkowski/src/kamera/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/home/nvidia/catkin_ws/src/pimot_szwajkowski/src/kamera/haarcascade_eye_tree_eyeglasses.xml')
    #body_cascade = cv2.CascadeClassifier('/home/nvidia/catkin_ws/src/pimot_szwajkowski/src/kamera/haarcascade_fullbody.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 255, 255), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = mask[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

    cv2.imshow("Obraz", mask)
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
