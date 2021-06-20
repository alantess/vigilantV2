import numpy as np
import cv2 as cv
import open3d as o3d
from generate_cloud import FeatureExtractor

W, H = 720, 480
VIDEO = '../../etc/videos/driving.mp4'


class Display(object):
    def __init__(self):
        # Gui Settings for Point Cloud
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=W, height=H, top=600, left=650)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        ctr = self.vis.get_view_control()
        # ctr.set_lookat([0, 1, 0])
        ctr.set_front([1, 0, 0])
        # ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.25)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1.5)

        self.geometry = o3d.geometry.PointCloud()
        self.vis.add_geometry(frame)
        self.vis.clear_geometries()
        self.vis.add_geometry(self.geometry)
        self.extractor = FeatureExtractor(H, W)
        # Open Video
        self.cap = cv.VideoCapture(VIDEO)

    def show(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            frame = cv.resize(frame, (W, H))
            img, xyz = self.extractor.extract(frame)
            self.display_lidar(xyz)
            cv.imshow('frame', img)
            if cv.waitKey(1) == ord('q'):
                break

        self.vis.run()
        self.vis.destroy_window()
        cv.destroyAllWindows()

    def display_lidar(self, xyz):
        self.geometry.points = o3d.utility.Vector3dVector(xyz)
        self.vis.update_geometry(self.geometry)
        self.vis.update_renderer()
        self.vis.poll_events()


if __name__ == '__main__':
    display = Display()
    display.show()
