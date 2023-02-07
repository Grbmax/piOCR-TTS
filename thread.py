import cv2
import pytesseract
import numpy as np
from pytesseract import Output
import threading

def speak(text):
    print("Speaking: ", text)

def get_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def blur_detect(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def extract_text_from_image(gray):
    t = pytesseract.image_to_string(gray)
    if t and t.strip() != "":
        print("Detected Text: ", t)
        tts_thread = threading.Thread(target=speak, args=(t,))
        tts_thread.start()
    else :
        print("No text detected in the image")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = get_gray(frame)
    bd = blur_detect(gray)

    if bd < 100:
        print("Image too blurry, try again")
    else:
        extract_text_from_image(gray)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
