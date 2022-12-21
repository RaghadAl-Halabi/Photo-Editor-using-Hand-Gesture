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

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv.LUT(src, table)

def nothing(x):
    pass

def setMaskTrackbar():
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



def histogram(img):
    # roi = firstFrame[r:r+h, c:c+w]

    yl = cv.getTrackbarPos("Y Low", "Settings")
    yu = cv.getTrackbarPos("Y Up", "Settings")
    crl = cv.getTrackbarPos("Cr Low", "Settings")
    cru = cv.getTrackbarPos("Cr Up", "Settings")
    cbl = cv.getTrackbarPos("Cb Low", "Settings")
    cbu = cv.getTrackbarPos("Cb Up", "Settings")

    low = np.array([yl, crl, cbl])
    up = np.array([yu, cru, cbu])

    mask = cv.inRange(img, low, up)
    ker = np.ones((5, 5), np.uint8)

    close = cv.morphologyEx(mask, cv.MORPH_CLOSE, ker)
    dilate = cv.dilate(close, ker, iterations=1)
    open = cv.morphologyEx(dilate, cv.MORPH_OPEN, ker)
    # Bilateral filtering in order to smooth the frame with the preserving of the edges
    mask = cv.bilateralFilter(open, 5, 50, 100)
    res = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow("Original", img)
    cv.imshow("Filter", res)

    low = np.array([yl, crl, cbl])
    up = np.array([yu, cru, cbu])
    mask = cv.inRange(img, low, up)
    cv.imshow("mask", mask)
    hist = cv.calcHist([firstFrame], [0], mask, [180], [0,180])
    cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
    return [hist], mask

def getMergedImageAfterEditingY(img_YCrCb):

    # Splitting the channels in order to deal only with the y channel that represents the illumination
    y, cr, cb = cv.split(img_YCrCb)
    #     cv.imshow("y", y)

    # Equalizing histogram of the y channel to effectively spread out the most frequent intensity values of illumination
    image_equ = cv.equalizeHist(y)
    # cv.imshow("Image_equalized", Image_equ)

    # Getting back the image in the YCrCb space
    image_merge = cv.merge([image_equ, cr, cb])
    # cv.imshow('image_merge', image_merge)
    return image_merge

setMaskTrackbar()

while(1):
    ret, frame = camera.read()

    frame = cv.flip(frame, 1)

    # Converting from gbr to YCbCr color space
    img_YCrCb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    # cv.imshow("img_YCrCb", img_YCrCb)

    image_merge = getMergedImageAfterEditingY(img_YCrCb)

    [hist], dst = histogram(image_merge)

    # dst = cv.calcBackProject([image_merge], [0], hist, [0, 180], 1)

    '''''
    ForegroundBackground model
    '''''
    # cv.imshow("y", y)

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

    thresholdedSkin = cv.adaptiveThreshold(dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv.THRESH_BINARY_INV, 15, 4)
    cv.imshow('thresholdedSkin', thresholdedSkin)
    #
    # # Applying MORPH_OPEN then MORPH_CLOSE to enhance the skin mask
    res = cv.morphologyEx(thresholdedSkin, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    res = cv.morphologyEx(thresholdedSkin, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cv.imshow("res", res)


    #
    # FOR LATER USE
    # edges = cv.Canny(skin_mask, 100, 200)
    # cv.imshow('edges', edges)

    # Non-skin Removal based on the skin mask on the ForegroundBackground model
    skinForeground = cv.bitwise_or(dilation, res)
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
    contours, hierarchy = cv.findContours(skinForeground, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
                res = contours[ci]
                hull = cv.convexHull(res)
                drawing = np.zeros(frame.shape, np.uint8)
                cv.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                cv.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

                # isFinishCal, cnt = calculateFingers(res, drawing)
                # print
                # "Fingers", cnt
                cv.imshow('output', drawing)



    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
camera.release()
cv.destroyAllWindows()
