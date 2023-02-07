import cv2
import pytesseract
import numpy as np
import threading
from pytesseract import Output
from gtts import gTTS
import os
import pyglet

def get_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def blur_detect(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = "img_op.mp3"
    tts.save(filename)
    music = pyglet.media.load(filename, streaming=False)
    music.play()
    os.remove(filename)

def play_audio(text):
    t = threading.Thread(target=speak, args=(text,))
    t.start()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = get_gray(frame)
    bd = blur_detect(gray)

    if bd < 100:
        print("Image too blurry, try again")
        play_audio('Image too blurry, try again')

    if bd > 100 :
        t = pytesseract.image_to_string(gray)
        if t and t.strip() != "":
            print("Grayscale Image: ",t)
            play_audio(t)
        else : 
            print("Nothing Detected in Grayscale Image")

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
