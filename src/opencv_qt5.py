import cv2 as cv
import sys
from PyQt5.QtGui import QPixmap, qRed
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QRadioButton, QSlider
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QRect
from PyQt5.QtCore import Qt


class FaceApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.width = 900
        self.height = 640
        self.pos_x = 200
        self.pos_y = 200

        self.pick_canny = False
        self.pick_gray = False
        self.pick_face_recognition = True

        self.setWindowTitle("opencv-qt5_vision")
        self.setGeometry(self.pos_x, self.pos_y, self.width, self.height)

        # variables for vision
        self.n_neighbors = 5
        self.resolution = (640, 480)

        # label
        self.vid_label = QLabel(self)
        self.neighbor_label = QLabel(self)
        self.neighbors_value_label = QLabel(self)
        self.effect_label = QLabel(self)
        self.face_rec_label = QLabel(self)
        self.gray_scale_label = QLabel(self)
        self.edge_detection_label = QLabel(self)

        # slider
        self.slider_neighbors = QSlider(Qt.Horizontal, self)

        # radiobutton
        self.rb_facerec = QRadioButton(self)
        self.rb_grayscale = QRadioButton(self)
        self.rb_edge_detection = QRadioButton(self)

        # VideoCapture
        self.cap = cv.VideoCapture(0)
        self.change_resolution(self.cap, 640, 480)

        # Radiobutton on-toggled event
        self.rb_facerec.toggled.connect(self.rb_face_recognition_checked)
        self.rb_grayscale.toggled.connect(self.rb_grayscale_checked)
        self.rb_edge_detection.toggled.connect(
            self.rb_edge_detection_checked)

        # set User Interface
        self.set_ui()

    def set_ui(self):
        # video label
        self.vid_label.setText("Vid Label")
        self.vid_label.setGeometry(QRect(100, 0, 620, 650))

        # neighbor slider label
        self.neighbor_label.setText("Neighbors: ")
        self.neighbor_label.setGeometry(QRect(500, 200, 100, 40))

        # neighbor slider
        self.slider_neighbors.setGeometry(QRect(510, 240, 200, 30))
        self.slider_neighbors.setMinimum((1))
        self.slider_neighbors.setMaximum(6)
        self.slider_neighbors.setSliderPosition(self.n_neighbors)
        self.slider_neighbors.valueChanged.connect(self.check_value)

        # neighbor value label
        self.neighbors_value_label.setGeometry(QRect(510, 270, 200, 30))
        self.neighbors_value_label.setText(str(self.n_neighbors))

        self.effect_label.setGeometry(QRect(510, 300, 200, 30))
        self.effect_label.setText("Effect: ")

        self.rb_facerec.setGeometry(QRect(510, 330, 30, 30))
        self.rb_facerec.setChecked(True)
        self.rb_grayscale.setGeometry(QRect(510, 360, 30, 30))
        self.rb_edge_detection.setGeometry(QRect(510, 390, 30, 30))

        self.face_rec_label.setGeometry(QRect(540, 330, 200, 30))
        self.face_rec_label.setText("Self-Recognition")

        self.gray_scale_label.setGeometry(QRect(540, 360, 200, 30))
        self.gray_scale_label.setText("Gray scale")

        self.edge_detection_label.setGeometry(QRect(540, 390, 200, 30))
        self.edge_detection_label.setText("Edge Detection")

    def view_detect_cap(self):
        self.thread_is_active = True

        while self.thread_is_active:

            haar_cascade = cv.CascadeClassifier('src/haar_face.xml')

            ret, image = self.cap.read()
            image = self.rescale_size(image)
            image = cv.flip(image, 1)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            if self.pick_face_recognition:

                faces_rect = haar_cascade.detectMultiScale(
                    image, scaleFactor=1.1, minNeighbors=self.n_neighbors)

                for (x, y, w, h) in faces_rect:
                    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

                faces_rect = haar_cascade.detectMultiScale(
                    image, scaleFactor=1.1, minNeighbors=self.n_neighbors)

                for (x, y, w, h) in faces_rect:
                    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

            if self.pick_gray:
                image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

            if self.pick_canny:
                image = cv.Canny(image, 150, 175)

            if self.pick_gray or self.pick_canny:
                height, width = image.shape
                qImg = QImage(image.data, width, height,
                              width*3, QImage.Format_Grayscale16)
            else:
                height, width, channel = image.shape
                step = channel * width
                qImg = QImage(image.data, width, height,
                              step, QImage.Format_RGB888)

            self.vid_label.setPixmap(QPixmap.fromImage(qImg))

            if cv.waitKey(8) & 0xFF == ord('d'):
                self.thread_is_active = False

        self.cap.release()
        cv.destroyAllWindows()

    def check_value(self, val):
        self.neighbors_value_label.setText(str(self.slider_neighbors.value()))
        self.n_neighbors = self.slider_neighbors.value()

    def change_resolution(self, cap, width, height):
        cap.set(3, width)
        cap.set(4, height)

    def rescale_size(self, capture, scale_width=0.5, scale_height=0.75):
        height = int(capture.shape[0] * scale_height)
        width = int(capture.shape[1] * scale_width)
        # print(height, width)

        dimension = (height, width)

        return cv.resize(capture, dimension, interpolation=cv.INTER_AREA)

    def rb_face_recognition_checked(self, checked):
        if checked:
            self.pick_face_recognition = True

        if self.pick_canny or self.pick_gray:
            self.pick_gray = False
            self.pick_canny = False

    def rb_grayscale_checked(self, checked):
        if checked:
            self.pick_gray = True

        if self.pick_canny or self.pick_face_recognition:
            self.pick_face_recognition = False
            self.pick_canny = False

    def rb_edge_detection_checked(self, checked):
        if checked:
            self.pick_canny = True
            self.pick_gray = True

        if self.pick_face_recognition:
            self.pick_face_recognition = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = FaceApp()

    main_win.show()

    main_win.view_detect_cap()

    sys.exit(app.exec_())
