# rpi-OCR-TTS
Optical Character Recognition & Text to Speech with Blur Detection on Raspberry-Pi

1. Install Tesseract-OCR and gstreamer for pyglet
```
sudo apt install tesseract-ocr espeak
```
2. Install Python dependencies
```
pip3 install -r requirements.txt
```
3. Run on sample images
```
python3 image.py -i images/coffee.jpg
```
