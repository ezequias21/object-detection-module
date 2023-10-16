import requests
import cv2

class RequestApp():
    def __init__(self, api_url='http://localhost:3000/api/frames'):
        self.api_url = api_url 

    def send(self, img):
        _, jpeg = cv2.imencode('.jpg', img)
        jpegBytes = jpeg.tobytes()

        response = requests.post(self.api_url, files={'image': jpegBytes})