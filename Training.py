import cv2
import mediapipe as mp
import pandas as pd
import pyautogui
import time
import numpy as np
from sklearn.linear_model import LinearRegression
pyautogui.FAILSAFE = False
from pathlib import Path
import os
directory = os.getcwd()
FolderName = "\V2_50"

Folder = directory + FolderName
Path(Folder).mkdir(parents=True, exist_ok=True)


Source = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence= 0.5
    )
DisplayWidth, DisplayHeight = pyautogui.size() # Screen Size to scale up/down
RetrainScale = 1
Pause = 0
Calibration = 0     
CalibrationStart = 0
start = 0
end = 0
Retrain = False

LeftX1 = []
LeftX2 = []
LeftX3 = []
LeftX4 = []
LeftX5 = []

LeftY1 = []
LeftY2 = []
LeftY3 = []
LeftY4 = []
LeftY5 = []

RightX1 = []
RightX2 = []
RightX3 = []
RightX4 = []
RightX5 = []

RightY1 = []
RightY2 = []
RightY3 = []
RightY4 = []
RightY5 = []


LeftDia1 = []
LeftDia2 = []
LeftDia3 = []
LeftDia4 = []
LeftDia5 = []

RightDia1 = []
RightDia2 = []
RightDia3 = []
RightDia4 = []
RightDia5 = []


FaceX1 = []
FaceX2 = []
FaceX3 = []
FaceX4 = []
FaceX5 = []

FaceY1 = []
FaceY2 = []
FaceY3 = []
FaceY4 = []
FaceY5 = []

FaceMiddleX = []
FaceMiddleY = []
AverageEyeWidth = []
XRetrainInput = []
YRetrainInput = []

while True:
    _, Frame = Source.read()


    Frame = cv2.flip(Frame, 1) # Mirror Image
    RGBFrame = cv2.cvtColor(Frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(RGBFrame)
    landmark_points = output.multi_face_landmarks
    Frame_h, Frame_w, _ = Frame.shape     

    if Pause == 1:
        print ("Paused")

    elif landmark_points:

        landmarks = landmark_points[0].landmark
        MouthDiff = landmarks[87].y - landmarks[82].y

        Facex = int(landmarks[168].x * Frame_w)
        Facey = int(landmarks[168].y * Frame_h)
        if (Facex > 0):
            Facex = Facex
        else:
            Facex = 0
        if (Facey > 0):
            Facey = Facey
        else:
            Facex = 0
        cv2.circle(Frame, (Facex, Facey), 3, (255, 255, 0))

        Rightx = int(landmarks[473].x * Frame_w)
        Righty = int(landmarks[473].y * Frame_h)
        cv2.circle(Frame, (Rightx, Righty), 3, (0, 255, 0))

        Leftx = int(landmarks[468].x * Frame_w)
        Lefty = int(landmarks[468].y * Frame_h)
        cv2.circle(Frame, (Leftx, Lefty), 3, (0, 255, 0))

        DiaLeftx = int(landmarks[469].x * Frame_w)
        DiaLefty = int(landmarks[469].y * Frame_h)

        DiaLeftx_ = int(landmarks[471].x * Frame_w)
        DiaLefty_ = int(landmarks[471].y * Frame_h)

        DiaRightx = int(landmarks[474].x * Frame_w)
        DiaRighty = int(landmarks[474].y * Frame_h)

        DiaRightx_ = int(landmarks[476].x * Frame_w)
        DiaRighty_ = int(landmarks[476].y * Frame_h)
        
        if Leftx > 0 and Rightx > 0:
            Midx = int((Rightx + Leftx)/2)
        else:
            Midx = 0
        
        if Lefty > 0 and Righty > 0:
            Midy = int((Righty + Lefty)/2)
        else:
            Midy = 0
        cv2.circle(Frame, (Midx, Midy), 3, (255, 0, 0)) 

        if DiaLeftx > 0 and DiaLeftx_ > 0 and DiaLefty > 0 and DiaLefty_ > 0 :
            DiaLeft = int(100*((DiaLeftx - DiaLeftx_)^2 + (DiaLefty - DiaLefty_)^2)**0.5)
        else:
            DiaLeft = 0

        if DiaRightx > 0 and DiaRightx_ > 0 and DiaRighty > 0 and DiaRighty_ > 0 :
            DiaRight = int(100*((DiaRightx - DiaRightx_)^2 + (DiaRighty - DiaRighty_)^2)**0.5)
        else:
            DiaRight = 0

        Calibrater = np.zeros([DisplayHeight,DisplayWidth,3],dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        Calibrater.fill(255)
        cv2.putText(Calibrater, "Press c to start.", (100, int(DisplayHeight - 350)), font, 2, (0,0,0), 2)
        cv2.putText(Calibrater, "Try to sit in the center of frame.", (100, int(DisplayHeight - 250)), font, 2, (0,0,0), 2)
        cv2.putText(Calibrater, "Stare at red dot untill it turns green", (100, int(DisplayHeight - 150)) , font, 2, (0,0,0), 2)

        cv2.circle(Calibrater, (100, 100), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (DisplayWidth - 100, 100), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, ( 100, DisplayHeight - 100), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (DisplayWidth - 100, DisplayHeight - 100), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(DisplayWidth/2), int(DisplayHeight/2)), 5, (0, 0, 0), 10)


        if (start == 0):
            end = 0
        else:
            end = time.time()
        if CalibrationStart == 1:
            start = time.time()
            CalibrationStart = 0
        diff = end*1000 - start*1000

        if diff < 2000 and diff > 1:
            cv2.circle(Calibrater, (100, 100), 5, (0, 0, 255), 20)
        elif  diff  > 2000 and diff < 5000:
            #Top Left
            cv2.circle(Calibrater, (100, 100), 5, (0, 0, 255), 20)
            if(diff> 2500):
                LeftX1.append(Leftx)
                LeftY1.append(Lefty)
                RightX1.append(Rightx)
                RightY1.append(Righty)
                LeftDia1.append(DiaLeft)
                RightDia1.append(DiaRight)
                FaceX1.append(Facex)
                FaceY1.append(Facey)
        elif diff > 5000 and diff < 8000:
            #Top Right
            cv2.circle(Calibrater, (100, 100), 5, (0, 255, 0), 20)
            cv2.circle(Calibrater, (DisplayWidth - 100, 100), 5, (0, 0, 255), 20)
            if(diff> 5500):
                LeftX2.append(Leftx)
                LeftY2.append(Lefty)
                RightX2.append(Rightx)
                RightY2.append(Righty)
                LeftDia2.append(DiaLeft)
                RightDia2.append(DiaRight)
                FaceX2.append(Facex)
                FaceY2.append(Facey)
        elif diff > 8000 and diff < 11000:
            #Bottom Left
            cv2.circle(Calibrater, (100, 100), 5, (0, 255, 0), 20)
            cv2.circle(Calibrater, (DisplayWidth - 100, 100), 5, (0, 255, 0), 20)
            cv2.circle(Calibrater, ( 100, DisplayHeight - 100), 5, (0, 0, 255), 20)
            if(diff> 8500):
                LeftX3.append(Leftx)
                LeftY3.append(Lefty)
                RightX3.append(Rightx)
                RightY3.append(Righty)
                LeftDia3.append(DiaLeft)
                RightDia3.append(DiaRight)
                FaceX3.append(Facex)
                FaceY3.append(Facey)
        elif diff > 11000 and diff < 14000:
            #Bottom Right
            cv2.circle(Calibrater, (100, 100), 5, (0, 255, 0), 20)
            cv2.circle(Calibrater, (DisplayWidth - 100, 100), 5, (0, 255, 0), 20)
            cv2.circle(Calibrater, ( 100, DisplayHeight - 100), 5, (0, 255, 0), 20)
            cv2.circle(Calibrater, (DisplayWidth - 100, DisplayHeight - 100), 5, (0, 0, 255), 20)
            if(diff> 11500):
                LeftX4.append(Leftx)
                LeftY4.append(Lefty)
                RightX4.append(Rightx)
                RightY4.append(Righty)
                LeftDia4.append(DiaLeft)
                RightDia4.append(DiaRight)
                FaceX4.append(Facex)
                FaceY4.append(Facey)
        elif diff > 14000 and diff < 17000:
            #Middle
            cv2.circle(Calibrater, (100, 100), 5, (0, 255, 0), 20)
            cv2.circle(Calibrater, (DisplayWidth - 100, 100), 5, (0, 255, 0), 20)
            cv2.circle(Calibrater, ( 100, DisplayHeight - 100), 5, (0, 255, 0), 20)
            cv2.circle(Calibrater, (DisplayWidth - 100, DisplayHeight - 100), 5, (0, 255, 0), 20)
            cv2.circle(Calibrater, ( int(DisplayWidth/2), int(DisplayHeight/2)), 5, (0, 0, 255), 20)
            if(diff> 14500):
                LeftX5.append(Leftx)
                LeftY5.append(Lefty)
                RightX5.append(Rightx)
                RightY5.append(Righty)
                FaceMiddleX.append(Facex) 
                FaceMiddleY.append(Facey)
                LeftDia5.append(DiaLeft)
                RightDia5.append(DiaRight)
                FaceX5.append(Facex)
                FaceY5.append(Facey)
        elif diff > 17000:
            #All Green | Breather before closing
            cv2.circle(Calibrater, (100, 100), 5, (0, 255, 0), 20)
            cv2.circle(Calibrater, (DisplayWidth - 100, 100), 5, (0, 255, 0), 20)
            cv2.circle(Calibrater, ( 100, DisplayHeight - 100), 5, (0, 255, 0), 20)
            cv2.circle(Calibrater, (DisplayWidth - 100, DisplayHeight - 100), 5, (0, 255, 0), 20)
            cv2.circle(Calibrater, ( int(DisplayWidth/2), int(DisplayHeight/2)), 5, (0, 255, 0), 20)

        cv2.imshow('Calibration Window', Calibrater)
        if(diff > 18000):
            Calibration = 1
            #cv2.destroyWindow(Calibrater)

            LeftXinput = np.concatenate((LeftX1, LeftX2, LeftX3, LeftX4, LeftX5))
            RightXinput = np.concatenate((RightX1, RightX2, RightX3, RightX4, RightX5))
            Xlist1 = [100]*len(LeftX1)
            Xlist2 = [DisplayWidth-100]*len(LeftX2)
            Xlist3 = [100]*len(LeftX3)
            Xlist4 = [DisplayWidth-100]*len(LeftX4)
            Xlist5 = [DisplayWidth/2]*len(LeftX5)
            Xval = np.concatenate((Xlist1, Xlist2, Xlist3, Xlist4, Xlist5))


            LeftYinput = np.concatenate((LeftY1, LeftY2, LeftY3, LeftY4, LeftY5))
            RightYinput = np.concatenate((RightY1, RightY2, RightY3, RightY4, RightY5))
            Ylist1 = [135]*len(LeftY1)
            Ylist2 = [135]*len(LeftY2)
            Ylist3 = [DisplayHeight-100+35]*len(LeftY3)
            Ylist4 = [DisplayHeight-100+35]*len(LeftY4)
            Ylist5 = [DisplayHeight/2+35]*len(LeftY5)
            Yval = np.concatenate((Ylist1, Ylist2, Ylist3, Ylist4, Ylist5))
            
            print(LeftDia1)
            print(LeftDia2)
            print(LeftDia3)
            print(LeftDia4)
            print(LeftDia5)
            LeftDiaInput = np.concatenate((LeftDia1, LeftDia2, LeftDia3, LeftDia4, LeftDia5))
            print(LeftDiaInput)
            RightDiaInput = np.concatenate((RightDia1, RightDia2, RightDia3, RightDia4, RightDia5))
            print(RightDiaInput)
            FaceXinput = np.concatenate((FaceX1, FaceX2, FaceX3, FaceX4, FaceX5))
            FaceYinput = np.concatenate((FaceY1, FaceY2, FaceY3, FaceY4, FaceY5))


            df = pd.DataFrame(list(zip(LeftXinput, LeftYinput, RightXinput, RightYinput, LeftDiaInput, RightDiaInput, FaceXinput, FaceYinput, Xval, Yval)), columns= ['LeftXInput', 'LeftYInput', 'RightXInput', 'RightYInput', 'LeftDiaInput', 'RightDiaInput', 'FaceXinput', 'FaceYinput', 'XVal', 'YVal'])
            df.to_csv(Folder + '/Training.csv')
    else:
        print("Unable to detect face")


    cv2.imshow('Eye Tracking Project', Frame)
    cv2.waitKey(1)
    k = cv2.waitKey(1) & 0xff

    if  k == 99:
        CalibrationStart = 1
        Caibration = 0
        start = 0
        print ("Key press")
    if  k == 27: # esc to quit
        break
    if k == 32:
        print("Spacebar Click")
        if Pause == 0:
            Pause = 1
        else:
            Pause = 0


Source.release() 
cv2.destroyAllWindows()