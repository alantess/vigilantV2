#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>

#undef slots
#include <torch/script.h>
#include <torch/torch.h>
#define slots Q_SLOTS

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  MainWindow(QWidget *parent = nullptr);
  ~MainWindow();
public Q_SLOTS:
  void display_camera();

private:
  Ui::MainWindow *ui;
};

torch::jit::Module load_model(std::string model_name);

#endif // MAINWINDOW_H
