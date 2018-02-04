import cv2
cam = cv2.VideoCapture(0)

face_detector=cv2.CascadeClassifier('face_cascade.xml')

tot_samples=20

name=raw_input('enter the name: ')
uniq_id = raw_input("enter the id: ")
samples=0


while(True):
    _,frame = cam.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_img, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)             
        samples=samples+1
        cv2.imwrite("data/"+str(name)+'-'+str(uniq_id)+'-'+ str(samples)+".jpg", gray_img[y:y+h,x:x+w] )
        print("\nSaving "+str(name)+'-'+str(uniq_id)+'-'+str(samples)+".jpg....")        
        cv2.imshow('facer',frame)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    
    elif samples>tot_samples-1:
        break
print("\nfinished") 
cam.release()
cv2.destroyAllWindows()
