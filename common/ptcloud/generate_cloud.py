import numpy as np
import cv2 as cv

VIDEO = "../../etc/videos/driving.mp4"

H, W = 1024, 512


class FeatureExtractor(object):
    def __init__(self):
        self.cap = cv.VideoCapture(VIDEO)
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
        res[0, 2] = W // 2
        res[1, 2] = H // 2
        print(res)
        return res

    def show(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            self._extract_and_display(frame)
            if cv.waitKey(1) == ord('q'):
                break
        cv.destroyAllWindows()

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
            np.linalg.norm(img, axis=2, ord=2).astype(np.uint8), 2500, 0.01, 3)
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)

        # pts = cv.KeyPoint_convert(kps)

        img = cv.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=0)

        cv.imshow('video', img)

    def lidar_generation(self, uv):
        # Requires nx3 dims
        # z = uv[0] --> Pixel Count
        # x = (uv[0] - cu) * uv[2] / f_u
        # y = (uv[1] - cv) * uv[2]  / fv
        # returns zx3 array
        pass

    def _reshape_pts(self, pts):
        mesh_x, mesh_y = np.meshgrid(pts, pts)
        z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
        z_norm = (z - z.min()) / (z.max() - z.min())
        xyz = np.zeros((np.size(mesh_x), 3))
        xyz[:, 0] = np.reshape(mesh_x, -1)
        xyz[:, 1] = np.reshape(mesh_y, -1)
        xyz[:, 2] = np.reshape(z_norm, -1)
        print(xyz.shape)
        return xyz


if __name__ == '__main__':
    feat = FeatureExtractor()

    feat.show()
