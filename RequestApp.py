import socket
import cv2

class RequestApp():
    def __init__(self):
        self.server_address = ('104.131.67.86', 2814) 
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, img):
        _, jpeg = cv2.imencode('.jpg', img)
        jpegBytes = jpeg.tobytes()
        if len(jpegBytes) < 65536:
            self.socket.sendto(jpegBytes, self.server_address)
