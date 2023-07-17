import numpy as np
import cv2



def discoverCameras(maxIndex=10):
    # checks the first 10 indexes.
    validIdx = []
    for idx in range(maxIndex):
        cap = cv2.VideoCapture(idx)
        if cap.read()[0]:
            validIdx.append(idx)
            cap.release()
    return validIdx

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
