#include "mainwindow.h"
#include "./ui_maindisplay.h"

#define DEFAULT_HEIGHT 720
#define DEFAULT_WIDTH 1280

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {

  ui->setupUi(this);
  connect(ui->camera_btn, SIGNAL(clicked()), this, SLOT(display_camera()));
  connect(ui->video_btn, SIGNAL(clicked()), this, SLOT(display_camera()));
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::display_camera() {

  cv::VideoCapture cap;
  cv::Mat frame;
  int deviceID = 0;
  int apiID = cv::CAP_ANY;
  QPushButton *b = qobject_cast<QPushButton *>(sender());
  if (b) {
    // Enable Camera / Webcam
    if (b == ui->camera_btn) {
      cap.open(deviceID, apiID);
      if (!cap.isOpened()) {
        std::cerr << "\nError: Cannot open camera\n";
      }

      // Set the dimentions 1280x720, Remove AutoFocus/Focus
      cap.set(cv::CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH);
      cap.set(cv::CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT);
      cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
      cap.set(cv::CAP_PROP_FOCUS, 0);

      ui->camera_btn->setText("Stop");
      ui->video_btn->setText("DISABLED");

      ui->video_btn->setEnabled(false);
      // Enable Video
    } else if (b == ui->video_btn) {
      cap.open("../../../etc/videos/driving.mp4");
      ui->video_btn->setText("Stop");
      ui->camera_btn->setText("DISABLED");

      ui->camera_btn->setEnabled(false);
      // Neither button has been clicked
    } else {
      std::cout << "No Button Selected.";
    }
  }

  if (!cap.isOpened()) {
    std::cerr << "\nError: Cannot open camera\n";
  }

  std::cout << "Press spacebar to terminate.\n";
  // Loop through video or camera

  // Load model
  torch::jit::script::Module module;
  try {
    module = load_model("traced_lanesNet.pt");
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
  }

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

    if (cv::waitKey(1) >= 0) {
      this->close();
      break;
    }
  }
}

// Loads torchscript Module
torch::jit::Module load_model(std::string model_name) {
  std::string directory = "../models/" + model_name;
  torch::jit::Module module = torch::jit::load(directory);
  module.to(torch::kCUDA);
  module.eval();
  std::cout << "Module Loaded: " << model_name << std::endl;
  return module;
}
