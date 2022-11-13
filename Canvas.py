import numpy as np
import cv2
import os
import time
import HTmodule as htm

folderPath = "c:\\Users\\HP\\Desktop\\PYHTON ML\\Innotech\\headerImg"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (51, 51, 225)

cap = cv2.VideoCapture(0)
cap.set(3, 1260)
cap.set(4, 720)
detector = htm.handDetector(detectionCon=0.85)

xp, yp = 0, 0
brushThickness = 15
eraserSize = 50
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # image read
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # find handLandmarks
    img = detector.findHands(img)
    lmList = detector.trackPos(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        # tip of index fingers
        x1, y1 = lmList[8][1:]

        # tip of middle finger
        x2, y2 = lmList[12][1:]

        # checking if any finger is up
        fingers = detector.fingerUp()
        # print(fingers)

        # if selectionMode(2 fingers up)
        if fingers[1] and fingers[2]:

            xp, yp = 0, 0

            # print('Selection MOde')
            if y1 < 125:
                if 0 < x1 < 250:
                    header = overlayList[0]
                    drawColor = (51, 51, 225)

                elif 300 < x1 < 550:
                    header = overlayList[1]
                    drawColor = (255, 60, 42)

                elif 600 < x1 < 850:
                    header = overlayList[2]
                    drawColor = (26, 255, 247)

                elif 900 < x1 < 1150:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # if drawingMode(1 finger up)

        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            # print("Drawing MOde")

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserSize)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserSize)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    #Converting image in to a Gray format
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInverse)
    img = cv2.bitwise_or(img, imgCanvas)



    # setting header image
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
