import numpy as np
import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
fgbg = cv2.createBackgroundSubtractorMOG2()


while(cap.isOpened()):
    ret, frame = cap.read()
    mk_frame = fgbg.apply(frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    frame[dst>0.01*dst.max()]=[0,0,255]
    

    


    if ret==True:

        out.write(frame)
        cv2.imshow('grey',gray)
        cv2.imshow('mk_frame',mk_frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
