#this program searches for a human face, it prints 'searching' unitil it finds a face then it post a window



import numpy as np
import cv2


def initial_win():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imwrite("img.jpg", frame)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    img = cv2.imread('img.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img =x,y,x+w,y+h

    return(img)

start = initial_win()
count = 0
while True:
    try:
        test = start[0]
        test = int(test)
        break
    except TypeError:
        print("looking for human")
        start = initial_win()
while type(test) != int:
        start = initial_win()
        count += 1
        print(count)
        print(start)


for x in start:
    r1 = start[0]
    h1 = start[1]
    c1 = start[2]
    w1 = start[3]
    
r,h,c,w = r1,h1,c1,w1    
track_window = (c,r,w,h)
print(start)

