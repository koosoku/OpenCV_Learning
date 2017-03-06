import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('D:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('D:\opencv\sources\data\haarcascades\haarcascade_eye_tree_eyeglasses.xml')

if face_cascade.empty():
  raise IOError('Unable to load the face cascade classifier xml file')

if eye_cascade.empty():
  raise IOError('Unable to load the eye cascade classifier xml file')

sunglasses = cv2.imread('images/sunglasses.png')

h_mask, w_mask = sunglasses.shape[:2]

cap = cv2.VideoCapture(0)
scaling_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    print("webcame shape: " + repr(frame.shape[:2]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eye_circles = eye_cascade.detectMultiScale(gray, 1.1, 3)
    for (x,y,w,h) in eye_circles:
        print("X: " + repr(x))
        print("Y: " + repr(y))
        print("W: " + repr(w))
        print("H: " + repr(h))
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0 , 0))
    cv2.imshow('Face Detector', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()