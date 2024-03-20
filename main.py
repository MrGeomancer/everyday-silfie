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
        os.rename(fr'dataset\{name}', fr'dataset\{i}.{name[-3:]}')
        i += 1
        # print (name)


def train_model():
    enc = []
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

    with open('my_encoding.pickle', 'wb') as file:
        file.write(pickle.dumps(data))

    return "[INFO] File my_encodings.pickle successfully created"


def detect_faces(img):
    # превращение фотографии в серую для работы с ней
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # подгрузка модели обнаружения лиц alt-2 работает получше
    # face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    # найти лица на фото. Чем меньше значения - чем больше лиц она найдет (очень много мусора)
    # faces = face.detectMultiScale(img_gray,1.001,4)
    # faces = face.detectMultiScale(img_gray,1.01,7)
    faces = face.detectMultiScale(img_gray, 1.1, 11)
    print(faces)  # выводит координаты (x,y,w,h)
    print(len(faces))  # выводит кол-во лиц

    # для каждого найденного лица
    for (x, y, w, h) in faces:
        if x < 50: x = 50  # если x находится близко к краю, то сделать его подальше
        if y < 50: y = 50  # если y находится близко к краю, то сделать его подальше
        face = img[y - 50:y + h + 50, x - 50:x + w + 50]
        cv2.imshow('ree', face)  # показать обрезанное лицо
        cv2.waitKey()  # не закрывать его
        cv2.imwrite('84.png', face)
        if recognize_face(face):  # если вырезанное лицо совпадает с моим
            print('Vitya')
        else:
            print('not vitya')


def recognize_face(face):
    # подгрузить датасет моих лиц
    data = pickle.loads(open('my_encoding.pickle', 'rb').read())
    
    face2_img = face_recognition.load_image_file(f'84.png')
    face2_enc = face_recognition.face_encodings(face2_img)[0]

    print('face2_img:', face2_img)
    print('face:', face)
    print('face2_img type:', type(face2_img))
    print('face type:', type(face))
    face_enc = face_recognition.face_encodings(face)[0]
    # print('face2_enc:', face2_enc)
    # print('face_enc:', face_enc)
    result = face_recognition.compare_faces(data['encoding'], face2_enc)
    if result[0]:
        print("Same person!")
    else:
        print("Another person!")

    return result[0]


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


##

# detect_faces(img)
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
    # загрузка фотографии для сравнения
    img = cv2.imread('82.png')

    # print(train_model(img_gray))
    detect_faces(img)
    # recognize_face(img)


if __name__ == '__main__':
    main()
