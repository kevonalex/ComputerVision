import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0) # creates video object

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
PrevTime = 0
CurrentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converts colour to RGB for input into Hands object

    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)

               # if id == 0:
                #    cv2.circle(img, (cx,cy),20, (255,0,255), cv2.FILLED) # Palm of the hand
                #elif id == 4:
                #    cv2.circle(img, (cx,cy),5, (0,0,255), cv2.FILLED) # Thumb
                #elif id == 8:
                #    cv2.circle(img, (cx,cy),5, (0,0,255), cv2.FILLED) # Index
                #elif id == 12:
                #    cv2.circle(img, (cx,cy),5, (0,0,255), cv2.FILLED) # Middle
                #elif id == 16:
                #    cv2.circle(img, (cx,cy),5, (0,0,255), cv2.FILLED) # Ring
                #elif id == 20:
                #    cv2.circle(img, (cx,cy),5, (0,0,255), cv2.FILLED) # Pinky

                mpDraw.draw_landmarks(img, handLms,mpHands.HAND_CONNECTIONS)


    CurrentTime = time.time()
    fps = 1/(CurrentTime-PrevTime)
    PrevTime = CurrentTime

    cv2.putText(img,str(int(fps)),(30,45),cv2.FONT_HERSHEY_PLAIN,1,
    (0,0,255),1)

    cv2.imshow("IMG_HAND_TRACK1", img)
    cv2.waitKey(1)
