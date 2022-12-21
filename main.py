import math

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy
from pynput.mouse import Button, Controller
import os

camera = cv.VideoCapture(0)

_,firstFrame = camera.read()

camera.set(10, 200)

fgbg = cv.createBackgroundSubtractorMOG2(0,50)

def nothing(x):
    pass

def setMaskTrackbar():
    cv.namedWindow("Settings")
    cv.resizeWindow("Settings", 640, 250)

    cv.createTrackbar("Y Low", "Settings", 0, 255, nothing)
    cv.createTrackbar("Y Up", "Settings", 0, 255, nothing)
    cv.createTrackbar("Cr Low", "Settings", 0, 255, nothing)
    cv.createTrackbar("Cr Up", "Settings", 0, 255, nothing)
    cv.createTrackbar("Cb Low", "Settings", 0, 255, nothing)
    cv.createTrackbar("Cb Up", "Settings", 0, 255, nothing)

    cv.setTrackbarPos("Y Low", "Settings", 0)
    cv.setTrackbarPos("Y Up", "Settings", 255)
    cv.setTrackbarPos("Cr Low", "Settings", 133)
    cv.setTrackbarPos("Cr Up", "Settings", 173)
    cv.setTrackbarPos("Cb Low", "Settings", 77)
    cv.setTrackbarPos("Cb Up", "Settings", 127)

def getMaskTrackbar():
    yl = cv.getTrackbarPos("Y Low", "Settings")
    yu = cv.getTrackbarPos("Y Up", "Settings")
    crl = cv.getTrackbarPos("Cr Low", "Settings")
    cru = cv.getTrackbarPos("Cr Up", "Settings")
    cbl = cv.getTrackbarPos("Cb Low", "Settings")
    cbu = cv.getTrackbarPos("Cb Up", "Settings")
    return yl, yu,crl,cru,cbl,cbu

def maskSkin(img,yl, yu, crl, cru, cbl, cbu):
    low = np.array([yl, crl, cbl])
    up = np.array([yu, cru, cbu])

    skinMask = cv.inRange(img, low, up)
    ker = np.ones((5, 5), np.uint8)

    skinMask = cv.morphologyEx(skinMask, cv.MORPH_CLOSE, ker)
    skinMask = cv.dilate(skinMask, ker, iterations=1)
    skinMask = cv.morphologyEx(skinMask, cv.MORPH_OPEN, ker)
    # Bilateral filtering in order to smooth the frame with the preserving of the edges
    skinMask = cv.bilateralFilter(skinMask, 5, 50, 100)

    cv.imshow("Original", img)
    cv.imshow("skinMask", skinMask)


    # hist = cv.calcHist([img], [0], histogramMask, [180], [0,180])
    # cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
    return skinMask

def getMergedImageAfterEditingY(img_YCrCb):

    # Splitting the channels in order to deal only with the y channel that represents the illumination
    y, cr, cb = cv.split(img_YCrCb)
    #     cv.imshow("y", y)

    # # Equalizing histogram of the y channel to effectively spread out the most frequent intensity values of illumination
    # image_equ = cv.equalizeHist(y)
    # # cv.imshow("Image_equalized", Image_equ)

    clahe = cv.createCLAHE(clipLimit=3)

    image_equ = clahe.apply(y)

    # Getting back the image in the YCrCb space
    image_merge = cv.merge([image_equ, cr, cb])
    # cv.imshow('image_merge', image_merge)
    return image_merge

def getMaxCon(rThresh):
    contours, hierarchy = cv.findContours(rThresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    max_i = 0
    max_area = -1
    for i in range(len(contours)):
        hand = contours[i]
        area_hand = cv.contourArea(hand)
        if area_hand > max_area:
            max_area = area_hand
            max_i = i
    try:
        hand = contours[max_i]
        hull = cv.convexHull(hand)
        drawing = np.zeros(frame.shape, np.uint8)
        cv.drawContours(drawing, [hand], 0, (0, 255, 0), 2)
        cv.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
        cv.imshow('output', drawing)
    except:
        contours = [0]
        hand = contours[0]

    return contours, hand

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
                    cv.circle(res, far, 3, (255,255,0), -1)

                cv.line(res, start, end, (0,255,255), 2)

        return defects, num_def, res

    except:
        defects = [0]
        num_def = 0

        return defects, num_def, res


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
        s = defects[:, 0][:, 0]

        x = np.array(max_con[s][:, 0][:, 0], dtype=np.float)
        y = np.array(max_con[s][:, 0][:, 1], dtype=np.float)

        xp = cv.pow(cv.subtract(x, cx), 2)
        yp = cv.pow(cv.subtract(y, cy), 2)

        dist = cv.sqrt(cv.add(xp, yp))
        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(max_con[farthest_defect][0])

        cv.line(res, (cx, cy), farthest_point, (0, 255, 255), 2)

        return farthest_point, res

    except:
        farthest_point = 0

        return farthest_point, res

setMaskTrackbar()

yl, yu, crl, cru, cbl, cbu = getMaskTrackbar()
beg = True
while(1):
    ret, frame = camera.read()

    frame = cv.flip(frame, 1)
    cv.imshow("original", frame)

    '''''
    ForegroundBackground model
    '''''

    # Converting from gbr to YCbCr color space
    img_YCrCb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    # cv.imshow("img_YCrCb", img_YCrCb)

    # Bilateral filtering in order to smooth the frame with the preserving of the edges
    image_equ =cv.split(getMergedImageAfterEditingY(img_YCrCb))[0]
    blur = cv.bilateralFilter(image_equ, 5, 50, 100)

    # Adaptive thresholding in order to give better results for videos with varying illumination
    thresholdedImg = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY_INV, 15, 3)
    # cv.imshow('thresholdedImg', thresholdedImg)

    # Background Removal based on the moving objects
    fgbg.setDetectShadows(False)
    fgmask = fgbg.apply(thresholdedImg , 0)
    foreground = cv.bitwise_and(thresholdedImg, thresholdedImg, mask=fgmask)
    # cv.imshow('foreground', foreground)

    # Removing the noise (the white dots that are considered as moving elements in the frames) with MORPH_OPEN
    opening = cv.morphologyEx(foreground, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # cv.imshow("opening", opening)
    # contours2, _ = cv.findContours(opening, cv.RETR_TREE,
    #                                cv.CHAIN_APPROX_SIMPLE)
    # openingc = cv.drawContours(opening, contours1, -1, (0, 255, 0), 1)
    # cv.imshow('openingc', openingc)

    # Adding more weight for the foreground with DILATION
    kernel = np.ones((9, 9), np.uint8)
    dilation = cv.dilate(opening, kernel, iterations=1)
    cv.imshow("dilation", dilation)
    # contours2, _ = cv.findContours(opening, cv.RETR_TREE,
    #                                cv.CHAIN_APPROX_SIMPLE)
    # dilationc = cv.drawContours(dilation, contours1, -1, (0, 255, 0), 1)
    # cv.imshow('dilationc', dilationc)

    '''''
        Skin model
    '''''


    image_merge = getMergedImageAfterEditingY(img_YCrCb)

    prevyl, prevyu, prevcrl, prevcru, prevcbl, prevcbu = yl, yu, crl, cru, cbl, cbu
    yl, yu, crl, cru, cbl, cbu = getMaskTrackbar()

    SkinMask = maskSkin(image_merge, yl, yu, crl, cru, cbl, cbu)

    skin = cv.bitwise_and(image_merge, image_merge, mask = SkinMask)
    cv.imshow("skin", skin)

    skin =  cv.cvtColor(skin, cv.COLOR_YCrCb2BGR)
    cv.imshow("bgrSkin", skin)
    skin =  cv.cvtColor(skin, cv.COLOR_BGR2GRAY)
    thresholdedSkin = cv.adaptiveThreshold( skin, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv.THRESH_BINARY_INV, 15, 4)
    # cv.imshow('thresholdedSkin', thresholdedSkin)

    # Applying MORPH_ODPEN then MORPH_CLOSE to enhance the skin mask
    thresholdedSkin = cv.morphologyEx(thresholdedSkin, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    thresholdedSkin = cv.morphologyEx(thresholdedSkin, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cv.imshow("thresholdedSkin", thresholdedSkin)


    #
    # FOR LATER USE
    # edges = cv.Canny(skin_mask, 100, 200)
    # # cv.imshow('edges', edges)

    # Non-skin Removal based on the skin mask on the ForegroundBackground model
    skinForeground = cv.bitwise_or(dilation, thresholdedSkin)
    cv.imshow('skinForeground', skinForeground)


    # contours1, _ = cv.findContours(skinForeground, cv.RETR_TREE,
    #                                cv.CHAIN_APPROX_SIMPLE)
    # skinForegroundc = cv.drawContours(skinForeground, contours1, -1, (0, 255, 0), 1)
    # cv.imshow('skinForegroundc', skinForegroundc)
    #
    # image = cv.resize(skinForeground, dsize=(480, 720),
    #                    )
    # cv.imshow('image', image)

    # Getting the contours and convex hull
    # skinMask1 = copy.deepcopy(image)
    contours, hand = getMaxCon(skinForeground)
    defects, num_def, res_frame = findFingers(frame, hand)
    cx, cy = centroid(hand)
    if np.all(contours[0] > 0):
        cv.circle(res_frame, (cx, cy), 5, (0, 255, 0), 2)
        cv.imshow("frameCopy",res_frame)
    else:
        pass

    farthest_point, res_frame = findFarPoint(res_frame, cx, cy, defects, hand)
    cv.imshow("farthest_point", res_frame)

    # isFinishCal, cnt = calculateFingers(res, drawing)
                # print
                # "Fingers", cnt
    #defects, num_def = findFingers(skinForeground, max_con)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
camera.release()
cv.destroyAllWindows()
