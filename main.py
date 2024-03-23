import cv2
import os
import face_recognition
import pickle
from PIL import Image



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
    cv2img = cv2.imread(img)
    # превращение фотографии в серую для работы с ней
    img_gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)

    # подгрузка модели обнаружения лиц alt-2 работает получше
    # face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    # найти лица на фото. Чем меньше значения - чем больше лиц она найдет (очень много мусора)
    # faces = face.detectMultiScale(img_gray,1.001,4)
    # faces = face.detectMultiScale(img_gray,1.01,7)
    faces = face.detectMultiScale(img_gray, 1.1, 11)
    print(faces)  # выводит координаты (x,y,w,h)
    print(len(faces))  # выводит кол-во лиц
    i=0
    # для каждого найденного лица
    for (x, y, w, h) in faces:
        if x < 50: x = 50  # если x находится близко к краю, то сделать его подальше
        if y < 50: y = 50  # если y находится близко к краю, то сделать его подальше
        face = cv2img[y - 50:y + h + 50, x - 50:x + w + 50]
        # cv2.imshow('ree', face)  # показать обрезанное лицо
        # cv2.waitKey()  # не закрывать его
        # face_name = f'{img}({i}).png'
        # i += 1
        # cv2.imwrite(face_name, face) # временно
        if recognize_face(face):  # если вырезанное лицо совпадает с моим
            print('Vitya')
            eyes = find_eyes(face)
            print(eyes)
            print(f"Середина глаз: x:{eyes[0][0]+x-50},y:{eyes[0][1]+y-50}, x:{eyes[1][0]+x-50},y:{eyes[1][1]+y-50}")
            cv2.circle(cv2img, center=(eyes[0][0]+x-50,eyes[0][1]+y-50), radius=5, color=[255, 255, 0])
            cv2.circle(cv2img, center=(eyes[1][0]+x-50,eyes[1][1]+y-50), radius=5, color=[255, 255, 0])
            cv2img = centre_eyes(eyes, img)
            break
        else:
            print('not vitya')
    cv2.imshow('ree', cv2img)  # показать обрезанное лицо
    cv2.waitKey()  # не закрывать его



def recognize_face(face):
    face2=cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # подгрузить датасет моих лиц
    data = pickle.loads(open('my_encoding.pickle', 'rb').read())
    # print('face2_img:', face2_img)
    # print('face:', face)
    # print('face2_img type:', type(face2_img))
    # print('face type:', type(face))
    face_enc = face_recognition.face_encodings(face2)[0]
    # print('face2_enc:', face2_enc)
    # print('face_enc:', face_enc)
    result = face_recognition.compare_faces(data['encoding'], face_enc)
    return result[0]


def find_eyes(cv2img):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
    # eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye_tree_eyeglasses.xml')
    img_gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(img_gray, 1.1, 11)
    # eyes = eye_cascade.detectMultiScale(img_gray,1.001,4)
    # eyes = eye_cascade.detectMultiScale(img_gray,1.01,7)
    eyes_centre=[]
    print(eyes)
    for (x, y, w, h) in eyes:
        cv2.rectangle(cv2img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        eyecentre = (int(x + w / 2), int(y + h / 2))
        # print(f'Середина глаза: {eyecentre}')
        eyes_centre.append(eyecentre)
        # cv2.circle(cv2img, center=eyecentre, radius=5, color=[0, 255, 0])
    # cv2.imshow('ree', cv2img)  # показать обрезанное лицо
    # cv2.waitKey()  # не закрывать его
    print('писька')
    return eyes_centre


def centre_eyes(eyes,img):
    global stock_eyes
    from math import tan, pi, atan,sqrt
    x1=eyes[1][0]
    y1=eyes[1][1]
    x2=eyes[0][0]
    y2=eyes[0][1]
    y_eyes = [eyes[0][1],eyes[1][1]]
    ygol=atan((y2-y1)/(x2-x1))
    tograd = (180/pi)

    # (h, w) = img.shape[:2]
    # center = (x1, y1)
    # M = cv2.getRotationMatrix2D(center, ygol*tograd, 1 )
    # img = cv2.warpAffine(img, M, (w, h))

    # cv2.imshow('re2e', img)  # показать обрезанное лицо
    # cv2.waitKey()  # не закрывать его


    im1 = Image.open(img)
    im1= im1.rotate(angle=ygol*tograd,center = (x1, y1))
    distance_between_eyes = abs(x2 - x1)
    shift_x = stock_eyes['x1'] - x1
    shift_y = stock_eyes['y1'] - y1
    im1 = im1.transform(im1.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y))
    im1.show()



    print('ygol:', ygol * tograd)
    print('ygol 45:',tan(1/1))
    print('ygol 45:',atan(1/1)* tograd)

    return img


stock_eyes = {'x1':1032,'y1':470,'x2':836,'y2':470}

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

    # img = 'stock.png'
    img = '92.png'
    # img = '52.jpg'

    # print(train_model(img_gray))
    detect_faces(img)
    # recognize_face(img)


if __name__ == '__main__':
    main()
