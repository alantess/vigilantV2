import numpy as np
import cv2 as cv

VIDEO = "../../etc/videos/driving.mp4"

W, H = 512, 1024


class FeatureExtractor(object):
    def __init__(self):
        self.cap = cv.VideoCapture(VIDEO)

        self.orb = cv.ORB_create()

    def show(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            self._extract_and_display(frame)
            if cv.waitKey(1) == ord('q'):
                break

    def _extract_and_display(self, img):
        img = cv.resize(img, (H, W))
        # Splits images in Half
        left = img[:, :int(H / 2)].astype(np.uint8)
        right = img[:, int(H / 2):].astype(np.uint8)
        # Brute-Force Matching with ORB Descriptors
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        kp1, des1 = self.orb.detectAndCompute(left, None)
        kp2, des2 = self.orb.detectAndCompute(right, None)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        img = cv.drawMatches(left,
                             kp1,
                             right,
                             kp2,
                             matches[:10],
                             None,
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # Extracts ORB
        feats = cv.goodFeaturesToTrack(
            np.linalg.norm(img, axis=2).astype(np.uint8), 1500, 0.01, 3)
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)
        img = cv.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=0)
        cv.imshow('video', img)


if __name__ == '__main__':
    feat = FeatureExtractor()
    feat.show()
