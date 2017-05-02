# Royce Kim

# Object Outlining

import cv2
import numpy as np

from os import path

import final as fin

IMG_FOLDER = "images"

"""
Bilateral filtering - http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
Filtering - http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html
Sobel - http://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm
Morphology - http://docs.opencv.org/2.4/doc/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html
Contours - http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
Contours Area - http://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
FindContours Ex - http://opencvexamples.blogspot.com/2013/09/find-contour.html
Thresholding - http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
"""

#For function parameter on trackbar creation. Does nothing
def nothing(x):
    pass

# GUI with trackbars
def main():
    imgName = raw_input("Enter image name: ")
    numIter = 1
    numCont = -1
    windowName = "Image"
    trackbars = "Parameters"
    legend = "Legend"

    #Different windows for UI elements
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.namedWindow(trackbars, cv2.WINDOW_NORMAL)
    cv2.namedWindow(legend, cv2.WINDOW_NORMAL)

    #Read image and prep
    img = cv2.imread(path.join(IMG_FOLDER, imgName))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 5, 150, 150)

    #Set window for legend UI
    legendImg = np.zeros((100, 500, 3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(legendImg, 'ThreshType: 0 = Binary, 1 = OTSU, 2 = Adaptive', (1, 20), font, 0.5, (255, 255, 255), 1, cv2.CV_AA)
    cv2.putText(legendImg, 'E: Erode & D: Dilate', (1, 40), font, 0.5, (255, 255, 255), 1, cv2.CV_AA)
    cv2.putText(legendImg, 'Press Q to quit', (1, 60), font, 0.5, (255, 255, 255), 1, cv2.CV_AA)
    cv2.imshow(legend, legendImg)

    #Create trackbars for each parameter able to manipulate
    #0 - 1 values are switches
    cv2.createTrackbar("R", trackbars, 0, 255, nothing)
    cv2.createTrackbar("G", trackbars, 0, 255, nothing)
    cv2.createTrackbar("B", trackbars, 0, 255, nothing)
    #Line thickness
    cv2.createTrackbar("Thickness", trackbars, 0, 10, nothing)

    #Binary, OTSU, Adaptive
    cv2.createTrackbar("ThreshType", trackbars, 0, 2, nothing)
    #Threshold value. Only for binary
    cv2.createTrackbar("Threshold", trackbars, 0, 255, nothing)

    #Sets morph operations erosion or dilation and how many iterations
    cv2.createTrackbar("0:E \n1:D", trackbars, 0, 1, nothing)
    cv2.createTrackbar("Iterations", trackbars, 0, 10, nothing)

    #Only shows largest contour
    cv2.createTrackbar("Largest", trackbars, 0, 1, nothing)

    #Turns on and off the process
    cv2.createTrackbar("Outline", trackbars, 0, 1, nothing)

    #Draw
    while(True):
        resultImg = img.copy()

        #Get all trackbar values
        r = cv2.getTrackbarPos("R", trackbars)
        g = cv2.getTrackbarPos("G", trackbars)
        b = cv2.getTrackbarPos("B", trackbars)
        contThickness = cv2.getTrackbarPos("Thickness", trackbars)

        whichTh = cv2.getTrackbarPos("ThreshType", trackbars)
        thresh = cv2.getTrackbarPos("Threshold", trackbars)

        whichMorph = cv2.getTrackbarPos("0:E \n1:D", trackbars)
        numIter = cv2.getTrackbarPos("Iterations", trackbars)

        largestBool = cv2.getTrackbarPos("Largest", trackbars)
        outline = cv2.getTrackbarPos("Outline", trackbars)
        contRGB = (b, g, r)

        #If process switch is on
        if (outline != 0):
            #Run process
            contours = fin.getContours(blur, whichTh, thresh, whichMorph, numIter)
            if (numCont != 0):
                #If largest contour switch is on
                if (largestBool == 1):
                    contours = contours[:1]
                else:
                    contours = contours[:numCont]
            #Display number of contours found
            numContours = "Number of contours: " + str(len(contours))
            print numContours
            cv2.putText(resultImg, numContours, (1, 20), font, 0.5, (255, 255, 255), 1, cv2.CV_AA)
            #Draw contours
            cv2.drawContours(resultImg, contours, -1, contRGB, contThickness)

        #If user presses 'q', quit
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        #Show image
        cv2.imshow(windowName, resultImg)
    #Save image
    cv2.imwrite(imgName, resultImg)
    #Close GUI windows
    cv2.destroyAllWindows()

"""
#Video?
def main():
    vidName = "bounce.flv"
    numIter = 0
    numCont = -1
    contThickness = 2
    contRGB = (0, 255, 0)

    cap = cv2.VideoCapture("images/bounce.flv")
    print cap.isOpened()

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        print ret

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 5, 150, 150)
        #ret, thresh_img = cv2.threshold(blur,91,255,cv2.THRESH_BINARY)
        #contours =  cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
        #for c in contours:
            #cv2.drawContours(frame, [c], -1, (0,255,0), 3)
        contours = fin.getContours(blur, numIter)
        if (numCont != 0):
            contours = contours[:numCont]
        print len(contours)
        cv2.drawContours(frame, contours, -1, contRGB, contThickness)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

#Image
#Finds edges along shadows
def main():
    imgName = "tetris.png"
    numIter = 0
    numCont = -1
    contThickness = 2
    contRGB = (0, 255, 0)

    img = cv2.imread(path.join(IMG_FOLDER, imgName))

    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #hsv[:, :, 2] = 200
    #rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)

    #SigmaSpace values should be 10 < s < 150
    blur = cv2.bilateralFilter(gray, 5, 150, 150)
    #blur = cv2.GaussianBlur(blur, (5, 5), 0)

    #edges = cv2.Canny(blur, 50, 200)

    contours = fin.getContours(blur, numIter)
    if (numCont != 0):
        contours = contours[:numCont]
    print len(contours)

    cv2.drawContours(img, contours, -1, contRGB, contThickness)
    cv2.imwrite(imgName, img)
    cv2.imshow(imgName, img)
    cv2.waitKey(0)
"""

if __name__ == "__main__":
    main()
