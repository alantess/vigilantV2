/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.1.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout;
    QFrame *frame;
    QLabel *logo;
    QLabel *camera;
    QPushButton *camera_btn;
    QPushButton *video_btn;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(800, 600);
        MainWindow->setMaximumSize(QSize(800, 600));
        MainWindow->setStyleSheet(QString::fromUtf8("QMainWindow{\n"
"	border-color: rgb(192, 200, 233);\n"
"	background-color: rgb(0, 43, 77);\n"
"}"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        verticalLayout = new QVBoxLayout(centralwidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        frame = new QFrame(centralwidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setStyleSheet(QString::fromUtf8("QFrame {\n"
"	background-color: rgb(0, 43, 77);\n"
"}"));
        frame->setFrameShape(QFrame::NoFrame);
        frame->setFrameShadow(QFrame::Raised);
        logo = new QLabel(frame);
        logo->setObjectName(QString::fromUtf8("logo"));
        logo->setGeometry(QRect(10, 10, 91, 31));
        logo->setMaximumSize(QSize(250, 150));
        logo->setPixmap(QPixmap(QString::fromUtf8("../../../etc/tiny_logo_light.png")));
        camera = new QLabel(frame);
        camera->setObjectName(QString::fromUtf8("camera"));
        camera->setGeometry(QRect(30, 50, 721, 451));
        camera->setMinimumSize(QSize(400, 200));
        camera->setMaximumSize(QSize(721, 451));
        camera->setStyleSheet(QString::fromUtf8("QLabel{\n"
"border-radius:20px;\n"
"}"));
        camera->setFrameShape(QFrame::NoFrame);
        camera->setFrameShadow(QFrame::Raised);
        camera->setPixmap(QPixmap(QString::fromUtf8("../../../etc/logo_dark.png")));
        camera->setAlignment(Qt::AlignCenter);
        camera_btn = new QPushButton(frame);
        camera_btn->setObjectName(QString::fromUtf8("camera_btn"));
        camera_btn->setGeometry(QRect(350, 10, 89, 25));
        camera_btn->setStyleSheet(QString::fromUtf8("QPushButton{\n"
"	background-color: rgb(4, 77, 69);\n"
"	border-radius:10px;\n"
"	padding-left:5px;\n"
"	padding-right:5px;\n"
"	font: 700 italic 11pt \"Kinnari\";\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"QPushButton::hover{\n"
"	background-color: rgb(71, 76, 179);\n"
"}"));
        camera_btn->setAutoDefault(true);
        video_btn = new QPushButton(frame);
        video_btn->setObjectName(QString::fromUtf8("video_btn"));
        video_btn->setGeometry(QRect(220, 10, 89, 25));
        video_btn->setStyleSheet(QString::fromUtf8("QPushButton{\n"
"	background-color: rgb(4, 77, 69);\n"
"	border-radius:10px;\n"
"	padding-left:5px;\n"
"	padding-right:5px;\n"
"	font: 700 italic 11pt \"Kinnari\";\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"QPushButton::hover{\n"
"	background-color: rgb(71, 76, 179);\n"
"}"));
        video_btn->setAutoDefault(true);

        verticalLayout->addWidget(frame);

        MainWindow->setCentralWidget(centralwidget);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        logo->setText(QString());
        camera->setText(QString());
        camera_btn->setText(QCoreApplication::translate("MainWindow", "Camera", nullptr));
        video_btn->setText(QCoreApplication::translate("MainWindow", "Video", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
