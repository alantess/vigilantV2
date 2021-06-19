import numpy as np
import cv2 as cv
import open3d as o3d
from sklearn.preprocessing import MinMaxScaler

VIDEO = "../../etc/videos/driving.mp4"

W, H = 1024, 512


class FeatureExtractor(object):
    def __init__(self):

        self.scaler = MinMaxScaler()
        self.cap = cv.VideoCapture(VIDEO)
        self.orb = cv.ORB_create()
        # Camera intrinsics and extrinsics
        self.P = self._calibrate()

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        # Gui Settings for Point Cloud
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=W, height=H, top=600, left=650)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        geometry = o3d.geometry.PointCloud()
        self.vis.add_geometry(geometry)

        # self.vis.run()

    def _calibrate(self):
        res = np.eye(3, k=0)
        res[0, 2] = W // 2
        res[1, 2] = H // 2
        return res

    def show(self):
        print("Press Q to start.")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            self._extract_and_display(frame)

            if cv.waitKey(1) == ord('q'):
                break
        cv.destroyAllWindows()
        self.vis.destroy_window()

    # Extracts features
    def _extract_and_display(self, img):
        img = cv.resize(img, (W, H))
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
            np.mean(img, axis=2).astype(np.uint8), 3500, 0.01, 3)
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)

        pts = cv.KeyPoint_convert(kps)
        self.lidar_generation(pts)

        img = cv.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=0)

        cv.imshow('video', img)

    def lidar_generation(self, pts, max_high=1):
        # Requires nx3 dims
        # z = uv[0] --> Pixel Count
        # x = (uv[0] - cu) * uv[2] / f_u
        # y = (uv[1] - cv) * uv[2]  / fv
        uv_depth = self._2d_to_3d(pts)
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        pts_3d_rect = self.scaler.fit_transform(pts_3d_rect)

        valid = (pts_3d_rect[:, 0] >= 0) & (pts_3d_rect[:, 2] < 1)
        pts_3d_rect = pts_3d_rect[valid]
        self.display_lidar(pts_3d_rect)

    def display_lidar(self, xyz):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        ctr = self.vis.get_view_control()
        ctr.rotate(10, 0.0, 0.0)

        self.vis.clear_geometries()
        self.vis.add_geometry(pcd)
        self.vis.run()

    def _2d_to_3d(self, pts):
        rows, cols = pts.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, pts])
        points = points.reshape((3, -1))
        points = points.T
        return points


if __name__ == '__main__':
    feat = FeatureExtractor()

    feat.show()
