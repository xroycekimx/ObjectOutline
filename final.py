# Royce Kim

# Object Outlining

import cv2
import numpy as np

#Gets contours of image
#Parameters: source image, threshold type, threshold value, morphological operation type, # of iterations
def getContours(img, whichThresh, threshold, whichMorph, it):
    #Different morphological kernels
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ellipKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    crossKernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    #Choose which threshold type is used
    thresh = img
    if (whichThresh == 0):
        ret, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    elif (whichThresh == 1):
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    #Choose which morphological operation is used
    operation = thresh
    if (whichMorph == 0):
        operation = cv2.erode(thresh, rectKernel, iterations = it)
    else:
        operation = cv2.dilate(thresh, rectKernel, iterations = it)

    #Fix gaps in object lines
    opening = cv2.morphologyEx(operation, cv2.MORPH_OPEN, rectKernel)
    #Fix holes in object lines
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, rectKernel)

    #Find contours
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Sort contours from largest -> smallest
    contours = sorted(contours, key = len, reverse = True)
    return contours
