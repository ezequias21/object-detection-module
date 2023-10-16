from jetson_inference import detectNet
import numpy as np
import jetson.utils
import cv2

class msSSD():
    def __init__(self, network, threshold):
        self.network = network
        self.threshold = threshold
        self.net = detectNet(self.network, self.threshold)
        self.colors = np.random.uniform(0,255,size=(96,3))

    def detect(self, img, display=False):
        imgCuda = jetson.utils.cudaFromNumpy(img)                
        detections = self.net.Detect(imgCuda, overlay="box,labels,conf")

        objects=[]
        for d in detections: 
            className = self.net.GetClassDesc(d.ClassID) 
            objects.append([className, d])

            if display:
                left,top,right,bottom = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
                cv2.putText(img, f'FPS: {self.net.GetNetworkFPS()}', (30,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 2)
                self.draw_box(img, left, top, right, bottom, className, self.colors[d.ClassID])
        return objects
    
    def draw_box(self, img, left, top, right, bottom, className, color):
        colorText = (255, 255, 255)
        cv2.rectangle(img,  (left+(right-left), top - 18), (left, top), color, cv2.FILLED)
        cv2.rectangle(img, (left, top), (right, bottom), color, 1)
        cv2.putText(img, f"{className}", (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorText, 1)

def main():
    cap = cv2.VideoCapture(0)
    myModel =  msSSD("ssd-mobilenet-v2", 0.5)
    while True:
        _, img = cap.read()
        objects = myModel.detect(img, True)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()