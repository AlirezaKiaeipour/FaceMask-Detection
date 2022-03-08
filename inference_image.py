import argparse
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--input_model",type=str,help="Please Enter path of input model/  Example: model.h5")
parser.add_argument("--input_image",type=str,help="Please Enter path of input image/  Example: image.jpg")
arg = parser.parse_args()

model = load_model(arg.input_model)
mpfacedetection = mp.solutions.face_detection
detector = mpfacedetection.FaceDetection()
frame = cv2.imread(arg.input_image)
frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
face = detector.process(frame_rgb)
if face.detections:  
  for index,detection in enumerate(face.detections):
    bounding_box = detection.location_data.relative_bounding_box
    height,width = frame.shape[:2]
    x ,y= int(bounding_box.xmin*width) ,int(bounding_box.ymin*height)
    w ,h = int(bounding_box.width*width) ,int(bounding_box.height*height)
    
    pred_frame = frame_rgb[y-30:y+h,x:x+w]
    pred_frame = cv2.resize(pred_frame,(224,224))
    pred_frame = pred_frame / 255.0
    pred_frame = pred_frame.reshape(1,224,224,3)

    predict = np.argmax(model.predict(pred_frame))
    predict = np.where(predict==0,"Mask","No Mask")
    if predict == "Mask":
      cv2.rectangle(frame,(x,y-30),(x+w,y+h),(0,255,0),4)
      cv2.putText(frame,str(predict),(x,y-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    else:
      cv2.rectangle(frame,(x,y-30),(x+w,y+h),(0,0,255),4)
      cv2.putText(frame,str(predict),(x,y-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2) 
    
  cv2.imshow("Face Mask",frame)
  cv2.waitKey(0)
