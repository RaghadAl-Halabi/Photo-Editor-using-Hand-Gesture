import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy
from pynput.mouse import Button, Controller
import os



camera = cv.VideoCapture(0)

camera.set(10, 200)

fgbg = cv.createBackgroundSubtractorMOG2(0,50)

while(1):
    ret, frame = camera.read()

    frame = cv.flip(frame, 1)
    cv.imshow("original", frame)

    # Converting from gbr to YCbCr color space
    img_YCrCb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    # cv.imshow("img_YCrCb", img_YCrCb)

    '''''
    ForegroundBackground model
    '''''
    # Splitting the channels in order to deal only with the y channel that represents the illumination
    y, cr, cb = cv.split(img_YCrCb)
    # cv.imshow("y", y)

    # Bilateral filtering in order to smooth the frame with the preserving of the edges
    blur = cv.bilateralFilter(y, 5, 50, 100)

    # Adaptive thresholding in order to give better results for videos with varying illumination
    thresholdedImg = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY_INV, 15, 3)
    cv.imshow('thresholdedImg', thresholdedImg)

    # Background Removal based on the moving objects
    fgbg.setDetectShadows(False)
    fgmask = fgbg.apply(thresholdedImg , 0)
    foreground = cv.bitwise_and(thresholdedImg, thresholdedImg, mask=fgmask)
    cv.imshow('foreground', foreground)

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
    # cv.imshow("dilation", dilation)
    # contours2, _ = cv.findContours(opening, cv.RETR_TREE,
    #                                cv.CHAIN_APPROX_SIMPLE)
    # dilationc = cv.drawContours(dilation, contours1, -1, (0, 255, 0), 1)
    # cv.imshow('dilationc', dilationc)

    '''''
        Skin model
    '''''
    # Equalizing histogram of the y channel to effectively spread out the most frequent intensity values of illumination
    Image_equ = cv.equalizeHist(y)
    # cv.imshow("Image_equalized", Image_equ)

    # Getting back the image in the YCrCb space
    image_merge = cv.merge([y, cr, cb])
    cv.imshow('image_merge', image_merge)

    # The values ranges that represent the skin colors in the YCrCb space
    skin_mask_mint = np.array((0, 133, 77))
    skin_mask_maxt = np.array((255, 173, 127))

    # Creating the skin mask using the previous ranges
    skin_mask = cv.inRange(image_merge, skin_mask_mint, skin_mask_maxt)
    thresholdedSkin = cv.adaptiveThreshold(skin_mask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv.THRESH_BINARY_INV, 15, 4)
    cv.imshow('thresholdedSkin', thresholdedImg)

    # Applying MORPH_OPEN then MORPH_CLOSE to enhance the skin mask
    skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cv.imshow('skin_mask', skin_mask)

    # # FOR LATER USE
    # edges = cv.Canny(skin_mask, 100, 200)
    # # cv.imshow('edges', edges)

    # Non-skin Removal based on the skin mask on the ForegroundBackground model
    skinForeground = cv.bitwise_and(dilation, skin_mask)
    cv.imshow('skinForeground', skinForeground)


    # contours1, _ = cv.findContours(skinForeground, cv.RETR_TREE,
    #                                cv.CHAIN_APPROX_SIMPLE)
    # skinForegroundc = cv.drawContours(skinForeground, contours1, -1, (0, 255, 0), 1)
    # cv.imshow('skinForegroundc', skinForegroundc)
    #
    image = cv.resize(skinForeground, dsize=(720, 720),
                       interpolation=cv.INTER_NEAREST_EXACT)
    cv.imshow('image', image)

    # Getting the contours and convex hull
    skinMask1 = copy.deepcopy(image)
    contours, hierarchy = cv.findContours(foreground, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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
# #
# #
# # # im_ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCR_CB)
# # #
# # # skin_ycrcb_mint = np.array((0, 133, 77))
# # # skin_ycrcb_maxt = np.array((255, 173, 127))
# # # skin_ycrcb = cv.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
# # #
# # # contours, _ = cv.findContours(skin_ycrcb, cv.RETR_EXTERNAL,
# # #         cv.CHAIN_APPROX_SIMPLE)
# # # for i, c in enumerate(contours):
# # #     area = cv.contourArea(c)
# # #     if area > 1000:
# # #         cv.drawContours(frame, contours, i, (255, 0, 0), 3)
# # # cv.imwrite(sys.argv[3], im)         # Final image
# #
# #
# #
#
# # hist = cv.calcHist([img_YCrCb], [0], None, [histSize], ranges, accumulate=False)
#     # cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
#     #
#     # backproj = cv.calcBackProject([img_YCrCb], [0], hist, ranges, scale=1)
#     #
#     # cv.imshow('BackProj', backproj)
#
#     # hand_hist = cv.calcHist([blur], [0], None, [256], [0, 256])
#     # res=cv.normalize(hand_hist, hand_hist, 0, 255, cv.NORM_MINMAX)
#     # Image_equ = cv.equalizeHist(y)
#     # cv.imshow("Image_equalized", Image_equ)
# #
# # Image_equ = cv.equalizeHist(y)
# # cv.imshow("Image_equalized", Image_equ)
# # elbow figure/curve
# # pp k clustering
# fuzzy clustering