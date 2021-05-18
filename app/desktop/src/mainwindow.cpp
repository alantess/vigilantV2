#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>

#include "./ui_maindisplay.h"
#include "mainwindow.h"

#define DEFAULT_HEIGHT 720
#define DEFAULT_WIDTH 1280

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {

  ui->setupUi(this);
  connect(ui->submit_btn, SIGNAL(clicked()), this, SLOT(display_image()));
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::display_image() {
  cv::Mat frame;
  cv::VideoCapture cap;
  int deviceID = 0;
  int apiID = cv::CAP_ANY;
  cap.open(deviceID, apiID);
  if (!cap.isOpened()) {
    std::cerr << "\nError: Cannot open camera\n";
  }

  // Set the dimentions 1280x720, Remove AutoFocus/Focus
  cap.set(cv::CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT);
  cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
  cap.set(cv::CAP_PROP_FOCUS, 0);

  std::cout << "Press spacebar to terminate.\n";
  ui->submit_btn->setText("Stop");
  for (;;) {
    cap.read(frame);
    if (frame.empty()) {
      std::cerr << "Error: Blank frame grabbed\n";
    }
    // Reisze image and Change to RGB
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    // OPENCV to QPixMap
    QImage imdisplay((uchar *)frame.data, frame.cols, frame.rows, frame.step,
                     QImage::Format_RGB888);
    // Display Camera image
    ui->camera->setPixmap(QPixmap::fromImage(imdisplay));

    if (cv::waitKey(5) >= 0)
      break;
  }
}
