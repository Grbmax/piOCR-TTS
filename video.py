import cv2
import pytesseract
import numpy as np
from pytesseract import Output
import multiprocessing as mp
from gtts import gTTS
import os
import pyglet
import time 

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

def process_image(gray):
    bd = blur_detect(gray)
    if bd < 100:
        print("Image too blurry, try again")
        speak('Image too blurry, try again')
    else:
        t = pytesseract.image_to_string(gray)
        if t and t.strip() != "":
            print("Grayscale Image: ",t)
            speak(t)
        else: 
            print("Nothing Detected in Grayscale Image")

def run_detection(q):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = get_gray(frame)
        q.put(gray)
        if cv2.waitKey(1) & 0xFF == ord('q'): #Press q to exit loop
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    q = mp.Queue()
    p = mp.Process(target=run_detection, args=(q,))
    p.start()
    while True:
        gray = q.get()
        process_image(gray)
