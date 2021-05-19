import cv2 as cv
import numpy as np


def show_video(url, model=None):
    cap = cv.VideoCapture(url)
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    video = "https://github.com/commaai/speedchallenge/raw/master/data/test.mp4"
    show_video(video)
    print("Done")
