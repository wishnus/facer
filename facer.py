import cv2
import numpy as np
from PIL import Image
import os
import sqlite3

recognizer = cv2.createLBPHFaceRecognizer()
face_detector = cv2.CascadeClassifier("face_cascade.xml");
#id_name_dict={}


cv2.namedWindow("FaceR",cv2.WINDOW_NORMAL)
cv2.resizeWindow("FaceR",300,300)

blank=cv2.imread('background.jpg')
display=cv2.imread('background.jpg')
cv2.rectangle(display,(100,45),(200,85),(255,0,0),-1)
cv2.putText(display, "TRAIN", (105,80) ,cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 4)
cv2.rectangle(display,(100,100),(200,140),(255,0,0),-1)
cv2.putText(display, "CAMERA", (102,133) ,cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 4)  
cv2.rectangle(display,(100,155),(200,195),(255,0,0),-1)
cv2.putText(display, "EXIT", (125,187) ,cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 0, 255), 4)  


def add_to_database(Id,name):
    con_var=sqlite3.connect("facer_database.db")
    #chech whether id exists
    cmd="select * from people_details where ID="+str(Id)
    cursor=con_var.execute(cmd)
    id_exists=0
    for row in cursor:
        id_exists=1
    if(id_exists==1):
        cmd="update people_details set name="+str(name)+" where ID="+str(Id)
    else:
        cmd="insert into people_details values (" +str(Id)+ "," + str(name)+")"
    con_var.execute(cmd)
    con_var.commit()
    con_var.close()

def get_from_database(Id):
    con_var=sqlite3.connect("facer_database.db")
    cmd="select * from people_details where ID="+str(Id)   
    cursor=con_var.execute(cmd)
    details=None
    for row in cursor:
        details=row
    con_var.close()
    return details



def train(uniq_id):
    cv2.imshow("FaceR",blank)    
    cam = cv2.VideoCapture(0)
    #face_detector=cv2.CascadeClassifier('face_cascade.xml')
    samples=0
    tot_samples=100
    while(True):
        _,frame=cam.read()
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        faces = face_detector.detectMultiScale(gray_img, 1.3, 5)    
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)             
            samples=samples+1
            cv2.imwrite("data/"+str(uniq_id)+'-'+ str(samples)+".jpg", gray_img[y:y+h,x:x+w] )
            print("\nSaving "+str(uniq_id)+'-'+str(samples)+".jpg....")        
            cv2.imshow('FaceR',frame)
        if samples>tot_samples-1:
            print("finished collecting data")
            #cv2.putText(blank, "Creating training ", (50,50) ,cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 4)
            #cv2.putText(blank, "data..... ", (70,75) ,cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 4) 
            cv2.imshow("FaceR",blank)
            cam.release()
            break
        cv2.waitKey(1)

    path='data'
    image_paths=[os.path.join(path,f) for f in os.listdir(path)]
    #face list
    face_samples=[]
    #id list
    ids=[]
    for image_path in image_paths:        
        pil_image=Image.open(image_path).convert('L')#converting to gray        
        image_data=np.array(pil_image,'uint8')#pil to numpy        
        _id=int(os.path.split(image_path)[-1].split("-")[0])
              
        faces=face_detector.detectMultiScale(image_data)        
        for (x,y,w,h) in faces:
            face_samples.append(image_data[y:y+h,x:x+w])
            ids.append(_id)
    ids=np.array(ids)
    print(ids)
    recognizer.train(face_samples, ids)
    recognizer.save('trained.yml')
    print("Trainer created...")
    cv2.imshow("FaceR",display)
    
    return 
    
        
    

def click_call(event,x,y,flags,param):
    #global id_name_dict
    if(event==cv2.EVENT_LBUTTONDOWN):
        #print("x:"+str(x)+" , y:"+str(y))
        if(x>100 and x<200):
            if(y>45 and y<85):
                print("Train")
                uniq_id=raw_input("Enter an unique id: ")
                name=raw_input("Enter the name: ")
                name='"'+str(name)+'"'
                add_to_database(uniq_id,name)                
                train(uniq_id)

                
                
            elif(y>100 and y<140):
                print("camera")
                recognizer.load('trained.yml')
                vid=cv2.VideoCapture(0)
                while True:
                    _, frame =vid.read()
                    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    faces=face_detector.detectMultiScale(gray_img,1.2,5)
                    for(x,y,w,h) in faces:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)
                        Id, surity = recognizer.predict(gray_img[y:y+h,x:x+w])
                        if(surity>40):
                            print("id->"+str(Id))
                            details=get_from_database(Id)
                            if(details!=None):
                                text=str(details[1])                    
                        else:
                            text="Unknown"
                                
                        cv2.putText(frame, text, (x,y+h),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 4)


                        
                    cv2.imshow('FaceR',frame)
                    if cv2.waitKey(1) & 0xFF==ord('q'):
                        cv2.imshow('FaceR',display)
                        break
                
            elif(y>155 and  y<195):
                print("exiting...")
                cv2.destroyAllWindows()
                




cv2.imshow('FaceR',display)

cv2.setMouseCallback('FaceR',click_call)

cv2.waitKey(0)
cv2.destroyAllWindows()
