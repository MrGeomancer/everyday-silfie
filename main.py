import cv2
import os
import numpy as np
import math
import face_recognition
import pickle


def dataset_get_ready():
    i = 0
    for name in os.listdir('dataset'):
        print(name)
        os.rename(fr'dataset\{name}',fr'dataset\{i}.{name[-3:]}')
        i += 1
        # print (name)


def train_model():
    enc=[]
    images = os.listdir('dataset')
    for image in images:
        face_img = face_recognition.load_image_file(f'dataset/{image}')
        face_enc = face_recognition.face_encodings(face_img)[0]
        if len(enc) == 0:
            enc.append(face_enc)
        else:
            for item in range(0, len(enc)):
                result = face_recognition.compare_faces([face_enc], enc[item])
                if result[0]:
                    enc.append(face_enc)
                    print("Same person!")
                    break
                else:
                    print("Another person!")
                    break


    data = {
        'name': 'victor',
        'encoding': enc
    }

    with open('my_encoding.pickle','wb') as file:
        file.write(pickle.dumps(data))

    return "[INFO] File my_encodings.pickle successfully created"


def detect_faces(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
    face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')

    # eyes = face.detectMultiScale(img_gray,1.001,4)
    # eyes = face.detectMultiScale(img_gray,1.01,7)
    faces = face.detectMultiScale(img_gray, 1.1, 11)
    print(faces)
    print(len(faces))
    for (x,y,w,h) in faces:
        face = img[y-20:y+h+20,x-20:x+w+20]
        # cv2.imshow('ree',face)
        # cv2.waitKey()
        if recognize_face(face):
            print('Vitya')
        else:
            print('not vitya')


def recognize_face(face):
    # data = pickle.loads(open('my_encoding.pickle', 'rb').read())
    picture_of_me = face_recognition.load_image_file("5.png")
    my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

    # unknown_picture = face_recognition.load_image_file(face)
    face_locations = face_recognition.face_locations(face)
    if len(face_locations) != 1:
        print('syka')
    top, right, bottom, left = face_locations[0]

    face_image = face[top:bottom, left:right]

    face_encodings = face_recognition.face_encodings(face_image)

    # Проверяем, что лицо было успешно закодировано
    if not face_encodings:
        print('net lica')

    # Берем первую кодировку лица
    new_face = face_encodings[0]

    # new_face = face_recognition.face_encodings(face)[0]
    # result = face_recognition.compare_faces(data['encoding'],new_face)
    result = face_recognition.compare_faces(my_face_encoding,new_face)
    return result



# # face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
# # face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye_tree_eyeglasses.xml')
# # face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')
# # face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
# face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')
# # face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_lefteye_2splits.xml')
# # face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_righteye_2splits.xml')
#
#
# # img = cv2.imread('stock.png')
img = cv2.imread('18.png')
##

detect_faces(img)
# # eyes = face.detectMultiScale(img_gray,1.001,4)
# eyes = face.detectMultiScale(img_gray,1.01,7)
# # eyes = face.detectMultiScale(img_gray, 1.1, 11)
# print(eyes)
# for (x, y, w, h) in eyes:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     eyecentre = (int(x + w / 2), int(y + h / 2))
#     print(f'Середина глаза: {eyecentre}')
#     cv2.circle(img, center=eyecentre, radius=5, color=[0, 255, 0])
#



def main():
    print(train_model(img_gray))
    detect_faces(img)



if __name__ == '__main__':
    main()