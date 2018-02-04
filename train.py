import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()
face_detector= cv2.CascadeClassifier("face_cascade.xml");


def getImagesAndIds(path):
    image_paths=[os.path.join(path,f) for f in os.listdir(path)] 
    #face list
    face_samples=[]
    #id list
    ids=[]
    
    for image_path in image_paths:
        
        pil_image=Image.open(image_path).convert('L')#converting to gray        
        image_data=np.array(pil_image,'uint8')#pil to numpy
        
        uniq_id=int(os.path.split(image_path)[-1].split("-")[1])
        #print(uniq_id)
        #name=os.path.split(image_path)[-1].split("-")[0]
        #print(name)
        
        faces=face_detector.detectMultiScale(image_data)
        
        for (x,y,w,h) in faces:
            face_samples.append(image_data[y:y+h,x:x+w])
            ids.append(uniq_id)
    ids=np.array(ids)
    return face_samples,ids

faces,ids = getImagesAndIds('data')
#print(faces)
#print(ids)
recognizer.train(faces, ids)
recognizer.save('trained.yml')
