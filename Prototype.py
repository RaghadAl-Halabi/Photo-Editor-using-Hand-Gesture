import cv2 as cv
import numpy as np
import os
import math
from pynput.mouse import Button, Controller

def takeTheFirstFrame():
    while (1):
        _, firstFrame = cap.read()
        firstFrame = cv.flip(firstFrame, 1)
        cv.imshow("firstFrame", firstFrame)
        k = cv.waitKey(1)
        if k & 0xFF == ord("o"):
            cv.destroyAllWindows()
            cv.imshow("fixedFirstFrame", firstFrame)
            while (1):
                k = cv.waitKey(1)
                if k & 0xFF == ord("r"):
                    return "r", firstFrame
                elif k & 0xFF == ord("d"):
                    return "d", firstFrame

def nothing(x):
    pass

def histogram(firstFrame, r, h, c, w):
    #roi = firstFrame[r:r+h, c:c+w]
    ycrcb_roi =  cv.cvtColor(firstFrame, cv.COLOR_BGR2YCrCb)

    # Splitting the channels in order to deal only with the y channel that represents the illumination
    y, cr, cb = cv.split(ycrcb_roi)
    #     cv.imshow("y", y)

    # Equalizing histogram of the y channel to effectively spread out the most frequent intensity values of illumination
    image_equ = cv.equalizeHist(y)
    # cv.imshow("Image_equalized", Image_equ)

    # Getting back the image in the YCrCb space
    image_merge = cv.merge([image_equ, cr, cb])
    # cv.imshow('image_merge', image_merge)

    yl, yu, crl, cru, cbl, cbu = setMask()
    low = np.array([yl, crl, cbl])
    up = np.array([yu, cru, cbu])
    mask = cv.inRange(ycrcb_roi, low, up)
    # cv.imshow("mask", mask)
    cv.waitKey(0)
    hist = cv.calcHist([ycrcb_roi], [0], mask, [180], [0,180])
    cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)

    return [hist]

def setMask():
    cv.namedWindow("Settings")
    cv.resizeWindow("Settings", 640, 250)

    cv.createTrackbar("Y Low", "Settings", 16, 235, nothing)
    cv.createTrackbar("Y Up", "Settings", 16, 235, nothing)
    cv.createTrackbar("Cr Low", "Settings", 16, 240, nothing)
    cv.createTrackbar("Cr Up", "Settings", 16, 240, nothing)
    cv.createTrackbar("Cb Low", "Settings", 16, 240, nothing)
    cv.createTrackbar("Cb Up", "Settings", 16, 240, nothing)

    cv.setTrackbarPos("Y Low", "Settings", 16)
    cv.setTrackbarPos("Y Up", "Settings", 235)
    cv.setTrackbarPos("Cr Low", "Settings", 16)
    cv.setTrackbarPos("Cr Up", "Settings", 240)
    cv.setTrackbarPos("Cb Low", "Settings", 16)
    cv.setTrackbarPos("Cb Up", "Settings", 240)

    while (1):
        _,frame = cap.read()
        frame = cv.flip(frame, 1)
        ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)

        yl = cv.getTrackbarPos("Y Low", "Settings")
        yu = cv.getTrackbarPos("Y Up", "Settings")
        crl = cv.getTrackbarPos("Cr Low", "Settings")
        cru = cv.getTrackbarPos("Cr Up", "Settings")
        cbl = cv.getTrackbarPos("Cb Low", "Settings")
        cbu = cv.getTrackbarPos("Cb Up", "Settings")
        
        low = np.array([yl, crl, cbl])
        up = np.array([yu, cru, cbu])

        mask = cv.inRange(ycrcb, low, up)
        ker = np.ones((5, 5), np.uint8)

        close = cv.morphologyEx(mask, cv.MORPH_CLOSE, ker)
        dilate = cv.dilate(close, ker, iterations = 1)
        open = cv.morphologyEx(dilate, cv.MORPH_OPEN, ker)
        # Bilateral filtering in order to smooth the frame with the preserving of the edges
        mask = cv.bilateralFilter(open, 5, 50, 100)
        res = cv.bitwise_and(frame, frame, mask = mask)
                
        cv.imshow("Original", frame)
        cv.imshow("Filter", res)

        k = cv.waitKey(1)
        if k & 0xFF == ord("s"):
            break
        
    cv.destroyAllWindows()
    
    return yl, yu, crl, cru, cbl, cru

def subtractBg(frame):
    fgMask = bgCap.apply(frame, learningRate = 0)
    ker = np.ones((3,3), np.uint8)
    fgMask = cv.erode(fgMask, ker, iterations = 1)
    res = cv.bitwise_and(frame, frame, mask = fgMask)
    
    return res

def findMaxContour(rThresh):
    con, _= cv.findContours(rThresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    max_i = 0
    max_area = 0
    for i in range(len(con)):
        hand = con[i]
        area_hand = cv.contourArea(hand)
        if area_hand > max_area:
            max_area = area_hand
            max_i = i
    try:
        max_con = con[max_i]
    except:
        con = [0]
        max_con = con[0]
        
    return con, max_con

def findFingers(res, max_con):
    try:
        hull = cv.convexHull(max_con, returnPoints = False)
        defects = cv.convexityDefects(max_con, hull)
        if defects is None:
            defects = [0]
            num_def = 0
        else:
            num_def = 0
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(max_con[s][0])
                end = tuple(max_con[e][0])
                far = tuple(max_con[f][0])
                
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                s = (a+b+c)/2
                ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
                
                d = (2*ar)/a
                
                angle = math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 57
                
                if angle <= 90 and d > 30:
                    num_def += 1
                    cv.circle(res, far, 3, (255,0,0), -1)
                    
                cv.line(res, start, end, (0,0,255), 2)
        
        return defects, num_def
    
    except:
        defects = [0]
        num_def = 0
        
        return defects, num_def

def centroid(max_con):
    moment = cv.moments(max_con)
    if moment is None:
        cx = 0
        cy = 0
        
        return cx, cy
    
    else:
        cx = 0
        cy = 0
        if moment["m00"] != 0:
            cx = int(moment["m10"] / moment["m00"])
            cy = int(moment["m01"] / moment["m00"])

        return cx, cy
    
def findFarPoint(res, cx, cy, defects, max_con):
    try:
        s = defects[:,0][:,0]

        x = np.array(max_con[s][:,0][:,0], dtype = np.float)
        y = np.array(max_con[s][:,0][:,1], dtype = np.float)

        xp = cv.pow(cv.subtract(x, cx), 2)
        yp = cv.pow(cv.subtract(y, cy), 2)
        
        dist = cv.sqrt(cv.add(xp, yp))
        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(max_con[farthest_defect][0])

        cv.line(res, (cx,cy), farthest_point, (0,255,255), 2)
        
        return farthest_point
        
    except:
        farthest_point = 0
        
        return farthest_point

def recognizeGestures(frame, num_def, count, farthest_point):
    try:
        print(num_def)
        # if num_def == 1:
        #     cv.putText(frame, "2", (0,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3, cv.LINE_AA)
        #     if count == 0:
        #         mouse.release(Button.left)
        #         mouse.position = (341, 82)
        #         mouse.press(Button.left)
        #         mouse.release(Button.left)
        #         mouse.position = farthest_point
        #         count = 1
        #
        # elif num_def == 2:
        #     cv.putText(frame, "3", (0,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3, cv.LINE_AA)
        #     if count == 0:
        #         mouse.release(Button.left)
        #         mouse.position = (254, 106)
        #         mouse.press(Button.left)
        #         mouse.release(Button.left)
        #         mouse.position = farthest_point
        #         count = 1
        #
        # elif num_def == 3:
        #     cv.putText(frame, "4", (0,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3, cv.LINE_AA)
        #     if count == 0:
        #         mouse.release(Button.left)
        #         mouse.position = (837, 69)
        #         mouse.press(Button.left)
        #         mouse.release(Button.left)
        #         mouse.position = farthest_point
        #         count = 1
        #
        # elif num_def == 4:
        #     cv.putText(frame, "5", (0,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3, cv.LINE_AA)
        #     if count == 0:
        #         mouse.release(Button.left)
        #         mouse.position = (772, 69)
        #         mouse.press(Button.left)
        #         mouse.release(Button.left)
        #         mouse.position = farthest_point
        #         count = 1
        #
        # else:
        #     cv.putText(frame, "1", (0,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3, cv.LINE_AA)
        #     mouse.position = farthest_point
        #     mouse.press(Button.left)
        #     count = 0

    except:
        print("You moved the hand too fast or take it out of range of vision of the camera")

cap = cv.VideoCapture(0)

v, firstFrame = takeTheFirstFrame()

while(1):
    if v == "r":
        cv.destroyAllWindows()
        v, firstFrame = takeTheFirstFrame()
    elif v == "d":
        cv.destroyAllWindows()
        break

r,h,c,w = 0,240,0,640
track_window = (c,r,w,h)
[hist] = histogram(firstFrame, r, h, c, w)
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

bgCaptured = False
cv.namedWindow("Value")
cv.resizeWindow("Value", 300, 60)
cv.createTrackbar("Thresh 1", "Value", 0, 20, nothing)
cv.createTrackbar("Thresh 2", "Value", 0, 20, nothing)

cv.setTrackbarPos("Thresh 1", "Value", 3)
cv.setTrackbarPos("Thresh 2", "Value", 3)


mouse = Controller()
count = 0
ex = 0

trigger = False
# OPEN SARAH'S GUI
# os.startfile("C:\Windows\system32\mspaint.exe")

while (1):
    ret,frame = cap.read()
    frame = cv.flip(frame, 1)

    ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    dst = cv.calcBackProject([ycrcb], [0], hist, [0,180], 1)
    ret,track_window = cv.CamShift(dst, track_window, term_crit)
    pts = cv.boxPoints(ret)
    pts = np.int0(pts)

    if bgCaptured is True:
        mask = subtractBg(frame)

        vThresh1 = cv.getTrackbarPos("Thresh 1", "Value")
        vThresh2 = cv.getTrackbarPos("Thresh 2", "Value")

        ker = np.ones((5,5), np.uint8)

        img = np.zeros(frame.shape, np.uint8)
        chanCount = mask.shape[2]
        ignoreColor = (255,) * chanCount
        cv.fillConvexPoly(img, pts, ignoreColor)
        res = cv.bitwise_and(mask, img)

        # Adding more weight for the foreground with DILATION
        resMask = cv.dilate(res, ker, iterations = 1)

        # Removing the noise (the white dots that are considered as moving elements in the frames) with MORPH_OPEN
        resMask = cv.morphologyEx(resMask, cv.MORPH_OPEN, ker)
        # Bilateral filtering in order to smooth the frame with the preserving of the edges
        resMask = cv.bilateralFilter(resMask, 5, 50, 100)
        resMask = cv.cvtColor(resMask, cv.COLOR_YCrCb2BGR)
        resMask = cv.cvtColor(resMask, cv.COLOR_BGR2GRAY)

        # Adaptive thresholding in order to give better results for videos with varying illumination
        rThresh = cv.adaptiveThreshold(resMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv.THRESH_BINARY_INV, vThresh1, vThresh2)

        con, max_con = findMaxContour(rThresh)

        defects, num_def = findFingers(res, max_con)

        cx, cy = centroid(max_con)
        if np.all(con[0] > 0):
            cv.circle(res, (cx,cy), 5, (0,255,0), 2)
        else:
            pass

        farthest_point = findFarPoint(res, cx, cy, defects, max_con)

        if trigger is True:
            recognizeGestures(frame, num_def, count, farthest_point)

        cv.imshow("Live", frame)
        cv.imshow("Result", res)
        cv.imshow("Threshold", rThresh)
        cv.imshow("Mask", mask)
        #cv.imshow("test", )

    k = cv.waitKey(1)
    if k & 0xFF == 27:
        break
    elif k == ord("c"):
        bgCap = cv.createBackgroundSubtractorMOG2(0,50)
        bgCaptured = True
    elif k == ord("a"):
        trigger = True

cv.destroyAllWindows()
# os.system("TASKKILL /F /IM c.exe")
cap.release()
