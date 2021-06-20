import numpy as np
import cv2 as cv
from sklearn.preprocessing import MinMaxScaler


class FeatureExtractor(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W
        self.scaler = MinMaxScaler()
        self.orb = cv.ORB_create()
        # Camera intrinsics and extrinsics
        self.P = self._calibrate()

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]

    def _calibrate(self):
        res = np.eye(3, k=0)
        res[0, 2] = self.W // 2
        res[1, 2] = self.H // 2
        return res

    # Extracts features
    def extract(self, img):
        # Splits images in Half
        left = img[:, :int(self.W / 2)].astype(np.uint8)
        right = img[:, int(self.W / 2):].astype(np.uint8)
        # Brute-Force Matching with ORB Descriptors
        bf = cv.BFMatcher(cv.NORM_HAMMING)

        kp1, des1 = self.orb.detectAndCompute(left, None)
        kp2, des2 = self.orb.detectAndCompute(right, None)

        matches = bf.knnMatch(des1, des2, 2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        img = cv.drawMatchesKnn(
            left,
            kp1,
            right,
            kp2,
            good,
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # Extracts ORB
        feats = cv.goodFeaturesToTrack(
            np.linalg.norm(img, axis=2, ord=2).astype(np.uint8), 2000, 0.01, 3)
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)

        pts = cv.KeyPoint_convert(kps)
        xyz = self.lidar_generation(pts)

        img = cv.drawKeypoints(img, kps, None, color=(248, 186, 255), flags=0)

        return img, xyz

    def lidar_generation(self, pts, max_high=0.5):
        uv_depth = self._2d_to_3d(pts)
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        pts_3d_rect = self.scaler.fit_transform(pts_3d_rect)

        valid = (pts_3d_rect[:, 0] >= 0) & (pts_3d_rect[:, 2] < max_high)
        pts_3d_rect = pts_3d_rect[valid]
        return pts_3d_rect

    def _2d_to_3d(self, pts):
        rows, cols = pts.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, pts])
        points = points.reshape((3, -1))
        points = points.T
        return points
