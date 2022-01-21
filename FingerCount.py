import cv2
import time
import os
import HandTracking as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = 'C:/Users/Michele/Documents/FingerImages'
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print("ciao Michelino, da Gioietta")
    if len(lmList) != 0:
        fingers = []
        if lmList[4][1] > lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        if lmList[8][2] < lmList[6][2]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(2,4):
            if lmList[tipIds[id]][2] < lmList[tipIds[id-2]][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        if lmList[20][2] < lmList[18][2]:
            fingers.append(1)
        else:
            fingers.append(0)

        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers].shape
        img[50:h+50, 50:w+50] = overlayList[totalFingers]
       # cv2.rectangle(img, (20, 225), (130, 325), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (25, 275), cv2.FONT_HERSHEY_PLAIN,
                    10, (0, 0, 0), 10)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
