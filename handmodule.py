import numpy as np
import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self,mode = False,maxHands=3, detectionCon = 0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        
        self.mphand = mp.solutions.hands
        self.hands = self.mphand.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mp_draw=mp.solutions.drawing_utils
        
    def findhands(self,img,draw=True):
        imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        self.results =  self.hands.process(imgrgb)

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand, self.mphand.HAND_CONNECTIONS)
        return img
            
    def findposition(self,img,handno=0,draw=True):
            
            lmlist=[]
            if self.results.multi_hand_landmarks:
                hand=self.results.multi_hand_landmarks[handno]
                for ids, lm in enumerate(hand.landmark):

                    h,w,c=img.shape
                    cx,cy = int(lm.x * w), int(lm.y* h)

                    lmlist.append([ids,cx,cy])
                    if draw:
                        cv2.circle(img , (cx,cy),5,(0,0,255),cv2.FILLED)
            return lmlist
   


def main():
    
    detector=HandDetector()
    cap=cv2.VideoCapture(0)
    while True:
        _,img= cap.read()
        img = cv2.flip(img,1)
        img = detector.findhands(img)
        lmlist =  detector.findposition(img)    
        cv2.imshow("",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    

if __name__=="__main__":
    main()