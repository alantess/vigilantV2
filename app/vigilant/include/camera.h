#include "iostream"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

typedef struct device {
  std::string device_type;
} device;

cv::Mat get_image(device *usr_device);
