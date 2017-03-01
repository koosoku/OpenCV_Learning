

import cv2

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

cur_char = -1
prev_char = -1

while True:
    # Read the current frame from webcam
    ret, frame = cap.read()

    # Resize the captured image
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    c = cv2.waitKey(1)

    if c == 27:
        break

    if c > -1 and c != prev_char and c!= 255:
        cur_char = c
    prev_char = c

    if cur_char == ord('g'):
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif cur_char == ord('y'):
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    elif cur_char == ord('h'):
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    else:
        output = frame

    cv2.imshow('Webcam', output)

cap.release()
cv2.destroyAllWindows()
