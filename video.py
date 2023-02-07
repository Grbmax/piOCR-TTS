import cv2
import pytesseract
import numpy as np
from pytesseract import Output
from gtts import gTTS
from time import sleep
import os
import pyglet

cap = cv2.VideoCapture(0)

def get_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def opening(image):
    kernel = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def canny(image):
    return cv2.Canny(image, 100, 200)

def blur_detect(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = "img_op.mp3"
    tts.save(filename)
    music = pyglet.media.load(filename, streaming=False)
    music.play()
    sleep(music.duration) #prevent from killing
    os.remove(filename)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.resize(frame, (640, 480))  # Reduce frame size
    gray = get_gray(frame)
    bd = blur_detect(gray)
    
    if bd < 100:
        print("Image too blurry, try again")
        speak('Image too blurry, try again')
        continue
    
    t = pytesseract.image_to_string(gray)
    if t and t.strip() != "":
        print("Grayscale Image: ", t)
        speak(t)
    else : 
        print("Nothing Detected in Grayscale Image")
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #Press q to exit loop
        break

cap.release()
cv2.destroyAllWindows()
