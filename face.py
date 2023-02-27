import cv2 as c
cascade=c.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = c.VideoCapture(0)
while True:
        _, img = cap.read()
        gray=c.cvtColor(img,c.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray,1.06,20)#
        for x ,y ,w ,h in faces:
            img=c.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)
        c.imshow("face detection",img)
        k=c.waitKey
        c.waitKey(0)
        k=c.waitKey(30)
        if k==27:
            break
        c.destroyAllWindows(0)