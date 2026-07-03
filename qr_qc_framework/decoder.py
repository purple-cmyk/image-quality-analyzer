
import cv2
det=cv2.QRCodeDetector()

def decode(img):
    txt,_,_=det.detectAndDecode(img)
    return txt!=""
