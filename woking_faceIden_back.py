import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def initial_win():
    ret, frame = cap.read()
    cv2.imwrite("img.jpg", frame)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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
        print('searching...')
        start = initial_win()
while type(test) != int:
        start = initial_win()
        count += 1
print('FOUND YOU >:b')
for x in start:
    r1 = start[0]
    h1 = start[1]
    c1 = start[2]
    w1 = start[3]
    
ret, frame = cap.read()

r,h,c,w = r1,h1,c1,w1    
track_window = c,r,w,h

roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, .5 )

while(1):
    ret, frame = cap.read()

    if ret==True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        cv2.imshow('img2',img2)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)


    else:
        break
    
cv2.destroyAllWindows()
cap.release()
