import cv2
import pytesseract
import numpy as np
from pytesseract import Output

def process_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t = pytesseract.image_to_string(gray)
    if t and t.strip() != "":
        print("Grayscale Image: ",t)
    else: 
        print("Nothing Detected in Grayscale Image")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    process_image(frame)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #Press q to exit loop
        break
cap.release()
cv2.destroyAllWindows()
