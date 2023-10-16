import cv2
import MobileNetModule as mnModule
import RequestApp
import argparse


parser = argparse.ArgumentParser(
                    prog='Detector de objetos',
                    description='Detecta objetos em imagens captadas por webcam',
                    epilog='')

parser.add_argument('-s', '--send-to-app',
                    dest='send_to_app',
                    action='store_true')

args = parser.parse_args()

cap = cv2.VideoCapture(0)
width =  cap.get(3)
height = cap.get(4)

cap.set(3, 640)
cap.set(4, 480)

myModel = mnModule.msSSD("ssd-mobilenet-v2", 0.5)
request = RequestApp.RequestApp()

while True:
    _, img = cap.read()
    myModel.detect(img, True)

    if args.send_to_app is True:
        request.send(img)

    cv2.imshow("Image", img)
    cv2.waitKey(1)