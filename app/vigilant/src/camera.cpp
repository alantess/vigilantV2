#include "camera.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>

cv::Mat get_image(device *usr_device) {
  cv::Mat frame;
  frame = cv::imread("../images/tiny_logo_light.png");
  return frame;
}
