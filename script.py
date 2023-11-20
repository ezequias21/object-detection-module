import cv2
import MobileNetModule as mnModule
import RequestApp
import argparse
import random
import os

parser = argparse.ArgumentParser(
                    prog='Detector de objetos',
                    description='Detecta objetos em imagens captadas por webcam',
                    epilog='')

parser.add_argument('-s', '--send-to-app',
                    dest='send_to_app',
                    action='store_true')

parser.add_argument('-r', '--render-frame',
                    dest='render_frame',
                    action='store_true')

args = parser.parse_args()

cap = cv2.VideoCapture(0)
width =  cap.get(3)
height = cap.get(4)

# cap.set(3, 640)
# cap.set(4, 480)

cap.set(3, 320)
cap.set(4, 240)

myModel = mnModule.msSSD("ssd-mobilenet-v2", 0.5)
request = RequestApp.RequestApp()

image_path=r"/home/estudante/Documentos/Estudo Jetson Nano/jetson-inference/build/aarch64/bin/backup"
os.chdir(image_path)

frame_identifier = 0

while True:
    _, img = cap.read()
    myModel.detect(img, True)

    if args.send_to_app is True:
        request.send(img)

    if args.render_frame is True:
        cv2.imshow("Image", img)

    frame_identifier+=1
    if (frame_identifier % 15) == 0:
        cv2.imwrite(f"image{random.randint(1, 10000)}.png", img)

    cv2.waitKey(1)
