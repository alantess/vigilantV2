#include "mainwindow.h"
#include "./ui_mainwindow.h"

#define DEFAULT_HEIGHT 720
#define DEFAULT_WIDTH 1280
#define IMG_SIZE 512

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
    module = load_model("quantized_lanesNet.pt");
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
  }
  auto start = std::chrono::high_resolution_clock::now();
  for (;;) {
    cap.read(frame);
    if (frame.empty()) {
      std::cerr << "Error: Blank frame grabbed\n";
    }
    // Change to RGB
    /* cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB); */
    // OPENCV to QPixMap
    frame = frame_prediction(frame, module);
    QImage imdisplay((uchar *)frame.data, frame.cols, frame.rows, frame.step,
                     QImage::Format_RGB888);
    // Display Camera image
    ui->camera->setPixmap(QPixmap::fromImage(imdisplay));

    if (cv::waitKey(1) >= 27) {
      break;
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "\nDuration: " << duration.count() << " microseconds\n";
}

// Loads torchscript Module
torch::jit::Module load_model(std::string model_name) {
  std::string directory = "../../../models/" + model_name;
  torch::jit::Module module = torch::jit::load(directory);
  /* module.to(torch::kCUDA); */
  module.eval();
  std::cout << "Module Loaded: " << model_name << std::endl;
  return module;
}

// Return a frame of the model prediction
cv::Mat frame_prediction(cv::Mat frame, torch::jit::Module model) {
  double alpha = 0.4;
  double beta = (1 - alpha);
  cv::Mat frame_copy, dst;
  std::vector<torch::jit::IValue> input;
  std::vector<double> mean = {0.406, 0.456, 0.485};
  std::vector<double> std = {0.225, 0.224, 0.229};
  cv::resize(frame, frame, cv::Size(IMG_SIZE, IMG_SIZE));
  frame_copy = frame;
  frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
  // Cv2 -> Tensor
  torch::Tensor frame_tensor =
      torch::from_blob(frame.data, {1, IMG_SIZE, IMG_SIZE, 3});
  frame_tensor = frame_tensor.permute({0, 3, 1, 2});
  frame_tensor = torch::data::transforms::Normalize<>(mean, std)(frame_tensor);
  /* frame_tensor = frame_tensor.to(torch::kCUDA); */
  input.push_back(frame_tensor);
  auto pred = model.forward(input).toTensor().detach().to(torch::kCPU);
  pred = pred.mul(100).clamp(0, 255).to(torch::kU8);

  cv::Mat output_mat(cv::Size{IMG_SIZE, IMG_SIZE}, CV_8UC1, pred.data_ptr());
  cv::cvtColor(output_mat, output_mat, cv::COLOR_GRAY2RGB);
  cv::applyColorMap(output_mat, output_mat, cv::COLORMAP_TWILIGHT_SHIFTED);

  cv::addWeighted(frame_copy, alpha, output_mat, beta, 0.0, dst);
  cv::resize(dst, dst, cv::Size(DEFAULT_WIDTH, DEFAULT_HEIGHT));
  return dst;
}
