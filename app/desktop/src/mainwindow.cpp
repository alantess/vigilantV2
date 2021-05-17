#include "mainwindow.h"
#include "./ui_maindisplay.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {

  ui->setupUi(this);
  connect(ui->submit_btn, SIGNAL(clicked()), this, SLOT(display_image()));
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::display_image() { ui->logo->setText("stfuuff"); }
