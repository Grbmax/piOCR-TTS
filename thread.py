import cv2
import pytesseract
import numpy as np
from pytesseract import Output
from gtts import gTTS
import threading
import time
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
    time.sleep(music.duration) #prevent from killing
    os.remove(filename)

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

vs = VideoStream().start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = cv2.resize(frame, (640, 480))
    gray = get_gray(frame)
    bd = blur_detect(gray)

    if bd < 100:
        print("Image too blurry, try again")
        speak('Image too blurry, try again')

    if bd > 100 :
        t = pytesseract.image_to_string(gray)
        if t and t.strip() != "":
            print("Grayscale Image: ",t)
            speak(t)
        else : 
            print("Nothing Detected in Grayscale Image")

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
