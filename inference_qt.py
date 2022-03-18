import sys
import argparse
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from PySide6.QtWidgets import QMainWindow , QApplication , QMessageBox
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QImage,QPixmap

parser = argparse.ArgumentParser()
parser.add_argument("--input_model",type=str,help="Please Enter path of input model/  Example: model.h5")
arg = parser.parse_args()

class Facemask(QMainWindow):
    def __init__(self):
        super().__init__()
        loader = QUiLoader()
        self.ui = loader.load("ui/form_cam.ui",None)
        self.ui.show()
        self.ui.btn_image.clicked.connect(self.screenshot)
        self.ui.info.triggered.connect(self.info)
        self.ui.exit.triggered.connect(exit)
        model = load_model(arg.input_model)

        mpfacedetection = mp.solutions.face_detection
        detector = mpfacedetection.FaceDetection()
        cap = cv2.VideoCapture(0)
        while True:
          _,self.frame = cap.read()
          self.frame_rgb = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)
          self.frame_rgb = cv2.resize(self.frame_rgb,(408,380))
          face = detector.process(self.frame_rgb)
          if face.detections:  
            for index,detection in enumerate(face.detections):
              self.bounding_box = detection.location_data.relative_bounding_box
              height,width = self.frame_rgb.shape[:2]
              x ,y= int(self.bounding_box.xmin*width) ,int(self.bounding_box.ymin*height)
              w ,h = int(self.bounding_box.width*width) ,int(self.bounding_box.height*height)
              
              pred_frame = self.frame_rgb[y:y+h,x:x+w]
              try:
                pred_frame = cv2.resize(pred_frame,(224,224))
                pred_frame = pred_frame / 255.0
                pred_frame = pred_frame.reshape(1,224,224,3)

                predict = np.argmax(model.predict(pred_frame))
                predict = np.where(predict==0,"Mask","No Mask")
                if predict == "No Mask":
                  cv2.rectangle(self.frame_rgb,(x,y),(x+w,y+h),(255,0,0),2)
                  cv2.putText(self.frame_rgb,str(predict),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                else:
                  cv2.rectangle(self.frame_rgb,(x,y),(x+w,y+h),(0,255,0),2)
                  cv2.putText(self.frame_rgb,str(predict),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2) 
              except:
                pass
          img = QImage(self.frame_rgb, self.frame_rgb.shape[1], self.frame_rgb.shape[0],QImage.Format_RGB888)
          pixmap = QPixmap.fromImage(img)
          self.ui.label_img.setPixmap(pixmap)
          self.key = cv2.waitKey(1)

    def screenshot(self):
      img = cv2.cvtColor(self.frame_rgb,cv2.COLOR_RGB2BGR)
      cv2.imwrite(f"photos/{self.bounding_box.xmin*self.bounding_box.height}.jpg",img)

    def info(self):
        msg = QMessageBox()
        msg.setText("Face Mask")
        msg.setInformativeText("GUI FaceMask Detection using Tensorflow & Keras\nThis program was developed by Alireza Kiaeipour\nContact developer: a.kiaipoor@gmail.com\nBuilt in 2022")
        msg.setIcon(QMessageBox.Information)
        msg.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Facemask()
    app.exec()