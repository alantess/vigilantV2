#include "mainwindow.h"
#include "./ui_maindisplay.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {

  ui->setupUi(this);
  ui->logo->setText("stfuuff");
}

MainWindow::~MainWindow() { delete ui; }
