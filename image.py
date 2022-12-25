import cv2 
import pytesseract
import numpy as np
from pytesseract import Output
from gtts import gTTS
from time import sleep
import os
import pyglet
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--window", type=bool, default=False, help="Show output window of OpenCv")
ap.add_argument("-i", "--images", required=True, help="Path to image to be detected")
ap.add_argument("-t", "--thresh", type=int, default=100, help="Focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

img_src = cv2.imread(args["images"])

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

gray = get_gray(img_src)
bd = blur_detect(gray)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)

if bd < args["thresh"]:
    print("Image too blurry, try again")
    speak('Image too blurry, try again')

if bd > args["thresh"] :    
     #Bounding Box Output
    if (args["window"] == True) :
        for img in [img_src, gray, thresh, opening, canny]:
            d = pytesseract.image_to_data(img, output_type=Output.DICT)
            n_boxes = len(d['text'])
            #Convert to RGB from Grayscale
            if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    
            for i in range(n_boxes):
                if int(d['conf'][i]) > 60:
                    (text, x, y, w, h) = (d['text'][i],d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    # don't show empty text
                if text and text.strip() != "":
                        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        img = cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.imshow('img', img)
                cv2.waitKey(0)
    if (args["window"] == False) :
        #Source Image
        t = pytesseract.image_to_string(img_src)
        if t and t.strip() != "":
            print("Source Image: ",t)
            speak(t)
        else : 
            print("Nothing Detected in Source Image")
        
        #Grayscale
        t = pytesseract.image_to_string(gray)
        if t and t.strip() != "":
            print("Grayscale Image: ",t)
            speak(t)
        else : 
            print("Nothing Detected in Grayscale Image")
        
        #Thresholding
        t = pytesseract.image_to_string(thresh)
        if t and t.strip() != "":
            print("Thresholded Image: ",t)
            speak(t)
        else : 
            print("Nothing Detected in Thresholded Image")
        
        #Morphed
        t = pytesseract.image_to_string(opening)
        if t and t.strip() != "":
            print("Morphed Image: ",t)
            speak(t)
        else : 
            print("Nothing Detected in Morphed Image")
        
        #Canny
        t = pytesseract.image_to_string(opening)
        if t and t.strip() != "":
            print("Morphed Image: ",t)
            speak(t)
        else : 
            print("Nothing Detected in Canny Image")
