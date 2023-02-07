import cv2
import pytesseract
import numpy as np
import multiprocessing as mp
from gtts import gTTS
import os
import pyglet
import threading

def get_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = "img_op.mp3"
    tts.save(filename)
    music = pyglet.media.load(filename, streaming=False)
    music.play()
    os.remove(filename)

def process_image(gray):
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
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #Press q to exit loop
            break
    cap.release()
    cv2.destroyAllWindows()

def run_tts(q):
    while True:
        gray = q.get()
        process_image(gray)

if __name__ == '__main__':
    q = mp.Queue()
    t1 = threading.Thread(target=run_detection, args=(q,))
    t2 = threading.Thread(target=run_tts, args=(q,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

