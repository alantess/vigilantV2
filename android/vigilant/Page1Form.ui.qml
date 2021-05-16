import QtQuick 2.12
import QtQuick.Controls 2.5

Page {
    width: 600
    height: 400

    Image {
        id: logo
        source: "images/tiny_logo_light.png"
        anchors {
            top: parent.top
            left: parent.left
        }
    }

    Image {
        id: camera
        source: ""
        anchors {
            centerIn: parent
        }
    }
}
