import cv2
import os
import numpy as np
import math

# face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
# face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye_tree_eyeglasses.xml')
# face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')
face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')
# face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_lefteye_2splits.xml')
# face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_righteye_2splits.xml')

def dataset_get_ready():
    i = 0
    for name in os.listdir('dataset'):
        print(name)
        os.rename(fr'dataset\{name}',fr'dataset\{i}.{name[-3:]}')
        i += 1
        # print (name)

# img = cv2.imread('stock.png')
img = cv2.imread('18.png')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# eyes = face.detectMultiScale(img_gray,1.001,4)
eyes = face.detectMultiScale(img_gray, 1.1, 11)
print(eyes)
for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    eyecentre = (int(x + w / 2), int(y + h / 2))
    print(f'Середина глаза: {eyecentre}')
    cv2.circle(img, center=eyecentre, radius=5, color=[0, 255, 0])

cv2.imshow('rez', img)
cv2.waitKey()
