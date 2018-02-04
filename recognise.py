import cv2
import numpy as np

recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trained.yml')
face_detector = cv2.CascadeClassifier("face_cascade.xml");


vid = cv2.VideoCapture(0)

while True:
    _, frame =vid.read()
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray_img,1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)
        Id, surity = recognizer.predict(gray_img[y:y+h,x:x+w])
        if(surity<50):
            text=str(Id)             
         
        else:
            text="Unknown"
        cv2.putText(frame, text, (x,y+h),cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 4)  
    cv2.imshow('FaceR',frame)
    
    if cv2.waitKey(5) & 0xFF==ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()
