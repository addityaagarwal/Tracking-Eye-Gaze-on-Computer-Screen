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
LeftX6 = []
LeftX7 = []
LeftX8 = []
LeftX9 = []
LeftX10 = []
LeftX11 = []
LeftX12 = []
LeftX13 = []
LeftX14 = []
LeftX15 = []
LeftX16 = []

LeftY1 = []
LeftY2 = []
LeftY3 = []
LeftY4 = []
LeftY5 = []
LeftY6 = []
LeftY7 = []
LeftY8 = []
LeftY9 = []
LeftY10 = []
LeftY11 = []
LeftY12 = []
LeftY13 = []
LeftY14 = []
LeftY15 = []
LeftY16 = []

RightX1 = []
RightX2 = []
RightX3 = []
RightX4 = []
RightX5 = []
RightX6 = []
RightX7 = []
RightX8 = []
RightX9 = []
RightX10 = []
RightX11 = []
RightX12 = []
RightX13 = []
RightX14 = []
RightX15 = []
RightX16 = []

RightY1 = []
RightY2 = []
RightY3 = []
RightY4 = []
RightY5 = []
RightY6 = []
RightY7 = []
RightY8 = []
RightY9 = []
RightY10 = []
RightY11 = []
RightY12 = []
RightY13 = []
RightY14 = []
RightY15 = []
RightY16 = []


LeftDia1 = []
LeftDia2 = []
LeftDia3 = []
LeftDia4 = []
LeftDia5 = []
LeftDia6 = []
LeftDia7 = []
LeftDia8 = []
LeftDia9 = []
LeftDia10 = []
LeftDia11 = []
LeftDia12 = []
LeftDia13 = []
LeftDia14 = []
LeftDia15 = []
LeftDia16 = []


RightDia1 = []
RightDia2 = []
RightDia3 = []
RightDia4 = []
RightDia5 = []
RightDia6 = []
RightDia7 = []
RightDia8 = []
RightDia9 = []
RightDia10 = []
RightDia11 = []
RightDia12 = []
RightDia13 = []
RightDia14 = []
RightDia15 = []
RightDia16 = []


FaceX1 = []
FaceX2 = []
FaceX3 = []
FaceX4 = []
FaceX5 = []
FaceX6 = []
FaceX7 = []
FaceX8 = []
FaceX9 = []
FaceX10 = []
FaceX11 = []
FaceX12 = []
FaceX13 = []
FaceX14 = []
FaceX15 = []
FaceX16 = []

FaceY1 = []
FaceY2 = []
FaceY3 = []
FaceY4 = []
FaceY5 = []
FaceY6 = []
FaceY7 = []
FaceY8 = []
FaceY9 = []
FaceY10 = []
FaceY11 = []
FaceY12 = []
FaceY13 = []
FaceY14 = []
FaceY15 = []
FaceY16 = []


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
        print("74")
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
        print("104")
        font = cv2.FONT_HERSHEY_SIMPLEX
        Calibrater.fill(255)
        cv2.putText(Calibrater, "Press c to start.", (100, int(DisplayHeight - 350)), font, 2, (0,0,0), 2)
        cv2.putText(Calibrater, "Try to sit in the center of frame.", (100, int(DisplayHeight - 250)), font, 2, (0,0,0), 2)
        cv2.putText(Calibrater, "Stare at red dot untill it turns green", (100, int(DisplayHeight - 150)) , font, 2, (0,0,0), 2)

        cv2.circle(Calibrater, (int(DisplayWidth/8), int(DisplayHeight/8)), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(3*DisplayWidth/8), int(DisplayHeight/8)), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(5*DisplayWidth/8), int(DisplayHeight/8)), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(7*DisplayWidth/8), int(DisplayHeight/8)), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(DisplayWidth/8), int(3*DisplayHeight/8)), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(3*DisplayWidth/8), int(3*DisplayHeight/8)), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(5*DisplayWidth/8), int(3*DisplayHeight/8)), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(7*DisplayWidth/8), int(3*DisplayHeight/8)), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(DisplayWidth/8), int(5*DisplayHeight/8)), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(3*DisplayWidth/8), int(5*DisplayHeight/8)), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(5*DisplayWidth/8), int(5*DisplayHeight/8)), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(7*DisplayWidth/8), int(5*DisplayHeight/8)), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(DisplayWidth/8), int(7*DisplayHeight/8)), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(3*DisplayWidth/8), int(7*DisplayHeight/8)), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(5*DisplayWidth/8), int(7*DisplayHeight/8)), 5, (0, 0, 0), 10)
        cv2.circle(Calibrater, (int(7*DisplayWidth/8), int(7*DisplayHeight/8)), 5, (0, 0, 0), 10)
        
        print("128")
        if (start == 0):
            end = 0
        else:
            end = time.time()
        if CalibrationStart == 1:
            start = time.time()
            CalibrationStart = 0
        diff = end*1000 - start*1000

        if diff < 2000 and diff > 1:
            cv2.circle(Calibrater, (int(DisplayWidth/8), int(DisplayHeight/8)), 5, (0, 0, 255), 10)
        elif  diff  > 2000 and diff < 5000:
            #1
            cv2.circle(Calibrater, (int(DisplayWidth/8), int(DisplayHeight/8)), 5, (0, 0, 255), 10)
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
            #2
            cv2.circle(Calibrater, (int(3*DisplayWidth/8), int(DisplayHeight/8)), 5, (0, 0, 255), 10)
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
            #3
            cv2.circle(Calibrater, (int(5*DisplayWidth/8), int(DisplayHeight/8)), 5, (0, 0, 255), 10)
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
            #4
            cv2.circle(Calibrater, (int(7*DisplayWidth/8), int(DisplayHeight/8)), 5, (0, 0, 255), 10)
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
            #5
            cv2.circle(Calibrater, (int(DisplayWidth/8), int(3*DisplayHeight/8)), 5, (0, 0, 255), 10)
            if(diff> 14500):
                LeftX5.append(Leftx)
                LeftY5.append(Lefty)
                RightX5.append(Rightx)
                RightY5.append(Righty)
                LeftDia5.append(DiaLeft)
                RightDia5.append(DiaRight)
                FaceX5.append(Facex)
                FaceY5.append(Facey)
        elif diff > 17000 and diff < 20000:
            #6
            cv2.circle(Calibrater, (int(3*DisplayWidth/8), int(3*DisplayHeight/8)), 5, (0, 0, 255), 10)
            if(diff> 17500):
                LeftX6.append(Leftx)
                LeftY6.append(Lefty)
                RightX6.append(Rightx)
                RightY6.append(Righty)
                LeftDia6.append(DiaLeft)
                RightDia6.append(DiaRight)
                FaceX6.append(Facex)
                FaceY6.append(Facey)
        elif diff > 20000 and diff < 23000:
            #7
            cv2.circle(Calibrater, (int(5*DisplayWidth/8), int(3*DisplayHeight/8)), 5, (0, 0, 255), 10)
            if(diff> 20500):
                LeftX7.append(Leftx)
                LeftY7.append(Lefty)
                RightX7.append(Rightx)
                RightY7.append(Righty)
                LeftDia7.append(DiaLeft)
                RightDia7.append(DiaRight)
                FaceX7.append(Facex)
                FaceY7.append(Facey)
        elif diff > 23000 and diff < 26000:
            #8
            cv2.circle(Calibrater, (int(7*DisplayWidth/8), int(3*DisplayHeight/8)), 5, (0, 0, 255), 10)
            if(diff> 23500):
                LeftX8.append(Leftx)
                LeftY8.append(Lefty)
                RightX8.append(Rightx)
                RightY8.append(Righty)
                LeftDia8.append(DiaLeft)
                RightDia8.append(DiaRight)
                FaceX8.append(Facex)
                FaceY8.append(Facey)
        elif diff > 26000 and diff < 29000:
            #9
            cv2.circle(Calibrater, (int(1*DisplayWidth/8), int(5*DisplayHeight/8)), 5, (0, 0, 255), 10)
            if(diff> 26500):
                LeftX9.append(Leftx)
                LeftY9.append(Lefty)
                RightX9.append(Rightx)
                RightY9.append(Righty)
                LeftDia9.append(DiaLeft)
                RightDia9.append(DiaRight)
                FaceX9.append(Facex)
                FaceY9.append(Facey)
        elif diff > 29000 and diff < 32000:
            #10
            cv2.circle(Calibrater, (int(3*DisplayWidth/8), int(5*DisplayHeight/8)), 5, (0, 0, 255), 10)
            if(diff> 29500):
                LeftX10.append(Leftx)
                LeftY10.append(Lefty)
                RightX10.append(Rightx)
                RightY10.append(Righty)
                LeftDia10.append(DiaLeft)
                RightDia10.append(DiaRight)
                FaceX10.append(Facex)
                FaceY10.append(Facey)
        elif diff > 32000 and diff < 35000:
            #11
            cv2.circle(Calibrater, (int(5*DisplayWidth/8), int(5*DisplayHeight/8)), 5, (0, 0, 255), 10)
            if(diff> 32500):
                LeftX11.append(Leftx)
                LeftY11.append(Lefty)
                RightX11.append(Rightx)
                RightY11.append(Righty)
                LeftDia11.append(DiaLeft)
                RightDia11.append(DiaRight)
                FaceX11.append(Facex)
                FaceY11.append(Facey)
        elif diff > 35000 and diff < 38000:
            #12
            cv2.circle(Calibrater, (int(7*DisplayWidth/8), int(5*DisplayHeight/8)), 5, (0, 0, 255), 10)
            if(diff> 17500):
                LeftX12.append(Leftx)
                LeftY12.append(Lefty)
                RightX12.append(Rightx)
                RightY12.append(Righty)
                LeftDia12.append(DiaLeft)
                RightDia12.append(DiaRight)
                FaceX12.append(Facex)
                FaceY12.append(Facey)
        elif diff > 38000 and diff < 41000:
            #13
            cv2.circle(Calibrater, (int(1*DisplayWidth/8), int(7*DisplayHeight/8)), 5, (0, 0, 255), 10)
            if(diff> 38500):
                LeftX13.append(Leftx)
                LeftY13.append(Lefty)
                RightX13.append(Rightx)
                RightY13.append(Righty)
                LeftDia13.append(DiaLeft)
                RightDia13.append(DiaRight)
                FaceX13.append(Facex)
                FaceY13.append(Facey)
        elif diff > 41000 and diff < 44000:
            #14
            cv2.circle(Calibrater, (int(3*DisplayWidth/8), int(7*DisplayHeight/8)), 5, (0, 0, 255), 10)
            if(diff> 41500):
                LeftX14.append(Leftx)
                LeftY14.append(Lefty)
                RightX14.append(Rightx)
                RightY14.append(Righty)
                LeftDia14.append(DiaLeft)
                RightDia14.append(DiaRight)
                FaceX14.append(Facex)
                FaceY14.append(Facey)
        elif diff > 44000 and diff < 47000:
            #15
            cv2.circle(Calibrater, (int(5*DisplayWidth/8), int(7*DisplayHeight/8)), 5, (0, 0, 255), 10)
            if(diff> 44500):
                LeftX15.append(Leftx)
                LeftY15.append(Lefty)
                RightX15.append(Rightx)
                RightY15.append(Righty)
                LeftDia15.append(DiaLeft)
                RightDia15.append(DiaRight)
                FaceX15.append(Facex)
                FaceY15.append(Facey)
        elif diff > 47000 and diff < 50000:
            #16
            cv2.circle(Calibrater, (int(7*DisplayWidth/8), int(7*DisplayHeight/8)), 5, (0, 0, 255), 10)
            if(diff> 47500):
                LeftX16.append(Leftx)
                LeftY16.append(Lefty)
                RightX16.append(Rightx)
                RightY16.append(Righty)
                LeftDia16.append(DiaLeft)
                RightDia16.append(DiaRight)
                FaceX16.append(Facex)
                FaceY16.append(Facey)
                
        cv2.imshow('Calibration Window', Calibrater)
        if(diff > 50000):
            Calibration = 1
            #cv2.destroyWindow(Calibrater)

            print (243)

            LeftXinput = np.concatenate((LeftX1, LeftX2, LeftX3, LeftX4, LeftX5, LeftX6, LeftX7, LeftX8, LeftX9, LeftX10, LeftX11, LeftX12, LeftX13, LeftX14, LeftX15, LeftX16))
            RightXinput = np.concatenate((RightX1, RightX2, RightX3, RightX4, RightX5, RightX6, RightX7, RightX8, RightX9, RightX10, RightX11, RightX12, RightX13, RightX14, RightX15, RightX16))
            Xlist1 = [int(DisplayWidth/8)]*len(LeftX1)
            Xlist2 = [int(3*DisplayWidth/8)]*len(LeftX2)
            Xlist3 = [int(5*DisplayWidth/8)]*len(LeftX3)
            Xlist4 = [int(7*DisplayWidth/8)]*len(LeftX4)
            Xlist5 = [int(DisplayWidth/8)]*len(LeftX5)
            Xlist6 = [int(3*DisplayWidth/8)]*len(LeftX6)
            Xlist7 = [int(5*DisplayWidth/8)]*len(LeftX7)
            Xlist8 = [int(7*DisplayWidth/8)]*len(LeftX8)
            Xlist9 = [int(DisplayWidth/8)]*len(LeftX9)
            Xlist10 = [int(3*DisplayWidth/8)]*len(LeftX10)
            Xlist11 = [int(5*DisplayWidth/8)]*len(LeftX11)
            Xlist12 = [int(7*DisplayWidth/8)]*len(LeftX12)
            Xlist13 = [int(DisplayWidth/8)]*len(LeftX13)
            Xlist14 = [int(3*DisplayWidth/8)]*len(LeftX14)
            Xlist15 = [int(5*DisplayWidth/8)]*len(LeftX15)
            Xlist16 = [int(7*DisplayWidth/8)]*len(LeftX16)
            Xval = np.concatenate((Xlist1, Xlist2, Xlist3, Xlist4, Xlist5, Xlist6, Xlist7, Xlist8, Xlist9, Xlist10, Xlist11, Xlist12, Xlist13, Xlist14, Xlist15, Xlist16))
            print (263)

            LeftYinput = np.concatenate((LeftY1, LeftY2, LeftY3, LeftY4, LeftY5, LeftY6, LeftY7, LeftY8, LeftY9, LeftY10, LeftY11, LeftY12, LeftY13, LeftY14, LeftY15, LeftY16))
            RightYinput = np.concatenate((RightY1, RightY2, RightY3, RightY4, RightY5, RightY6, RightY7, RightY8, RightY9, RightY10, RightY11, RightY12, RightY13, RightY14, RightY15, RightY16))
            Ylist1 = [int(DisplayHeight/8+35)]*len(LeftY1)
            Ylist2 = [int(DisplayHeight/8+35)]*len(LeftY2)
            Ylist3 = [int(DisplayHeight/8+35)]*len(LeftY3)
            Ylist4 = [int(DisplayHeight/8+35)]*len(LeftY4)
            Ylist5 = [int(3*DisplayHeight/8+35)]*len(LeftY5)
            Ylist6 = [int(3*DisplayHeight/8+35)]*len(LeftY6)
            Ylist7 = [int(3*DisplayHeight/8+35)]*len(LeftY7)
            Ylist8 = [int(3*DisplayHeight/8+35)]*len(LeftY8)
            Ylist9 = [int(5*DisplayHeight/8+35)]*len(LeftY9)
            Ylist10 = [int(5*DisplayHeight/8+35)]*len(LeftY10)
            Ylist11 = [int(5*DisplayHeight/8+35)]*len(LeftY11)
            Ylist12 = [int(5*DisplayHeight/8+35)]*len(LeftY12)
            Ylist13 = [int(7*DisplayHeight/8+35)]*len(LeftY13)
            Ylist14 = [int(7*DisplayHeight/8+35)]*len(LeftY14)
            Ylist15 = [int(7*DisplayHeight/8+35)]*len(LeftY15)
            Ylist16 = [int(7*DisplayHeight/8+35)]*len(LeftY16)
            Yval = np.concatenate((Ylist1, Ylist2, Ylist3, Ylist4, Ylist5, Ylist6, Ylist7, Ylist8, Ylist9, Ylist10, Ylist11, Ylist12, Ylist13, Ylist14, Ylist15, Ylist16))
            
            
            
            LeftDiaInput = np.concatenate((LeftDia1, LeftDia2, LeftDia3, LeftDia4, LeftDia5, LeftDia6, LeftDia7, LeftDia8, LeftDia9, LeftDia10, LeftDia11, LeftDia12, LeftDia13, LeftDia14, LeftDia15, LeftDia16))
            RightDiaInput = np.concatenate((RightDia1, RightDia2, RightDia3, RightDia4, RightDia5, RightDia6, RightDia7, RightDia8, RightDia9, RightDia10, RightDia11, RightDia12, RightDia13, RightDia14, RightDia15, RightDia16))
            FaceXinput = np.concatenate((FaceX1, FaceX2, FaceX3, FaceX4, FaceX5, FaceX6, FaceX7, FaceX8, FaceX9, FaceX10, FaceX11, FaceX12, FaceX13, FaceX14, FaceX15, FaceX16))
            FaceYinput = np.concatenate((FaceY1, FaceY2, FaceY3, FaceY4, FaceY5, FaceY6, FaceY7, FaceY8, FaceY9, FaceY10, FaceY11, FaceY12, FaceY13, FaceY14, FaceY15, FaceY16))

            df = pd.DataFrame(list(zip(LeftXinput, LeftYinput, RightXinput, RightYinput, LeftDiaInput, RightDiaInput, FaceXinput, FaceYinput, Xval, Yval)), columns= ['LeftXInput', 'LeftYInput', 'RightXInput', 'RightYInput', 'LeftDiaInput', 'RightDiaInput', 'FaceXinput', 'FaceYinput', 'XVal', 'YVal'])
            df.to_csv(Folder + '\Testing.csv')
            print ("Done")
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