import imghdr
import os
import cv2
import MobileNetModule as mnModule
import random


path_dir_base = '/home/estudante/Documentos/Estudo Jetson Nano/jetson-inference/build/aarch64/bin/ground_truth'
path_dir_base_bb = '/home/estudante/Documentos/Estudo Jetson Nano/jetson-inference/build/aarch64/bin/bounding_box'
thresholds = [0.4]

#person, car, motorcycle
classes = (1,3,4)

def open_image(threshold):
    myModel = mnModule.msSSD("ssd-mobilenet-v2", threshold)
  
    files = os.listdir(path_dir_base)
  
    if not os.path.exists(os.path.join(path_dir_base_bb, f'{threshold}')):
        os.makedirs(os.path.join(path_dir_base_bb, f'{threshold}'))

    for file in files:
        if file.lower().endswith(('.png', '.jpg')):
            image = cv2.imread(os.path.join(path_dir_base, file))
            detections = myModel.getDetections(image)
            
            width = image.shape[1]
            height = image.shape[0]
            textfilename = file.split('.')
           
            with open(os.path.join(path_dir_base_bb, f'{threshold}/{textfilename[0]}.txt'), 'w', encoding='utf-8') as textfile:
                for d in detections:
                    if d.ClassID in classes:
                        textfile.write(f'{d.ClassID} {d.Center[0]/width} {d.Center[1]/height} {d.Width/width} {d.Height/height}\n')
            exit()
            
def calcBoundingBoxs():
    for  threshold in thresholds: open_image(threshold)

calcBoundingBoxs()