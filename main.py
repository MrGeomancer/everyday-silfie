import cv2
import os
import face_recognition
import pickle
from PIL import Image, ImageTk

import tkinter as tk


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


def on_click2(event):
    global root2, vitya
    print(f'[{imgnum}] Было выбрано что это Витя')
    vitya = True
    root2.destroy()


def on_click3(event):
    global root2, vitya
    print(f'[{imgnum}] Было выбрано что это не Витя')
    vitya = False
    root2.destroy()


def on_click4(event):
    global root3, face_centre
    print(f'[{imgnum}] Было выбрано что это не Витя')
    face_centre = [event.x,event.y]
    root3.destroy()


def detect_faces(img,again):
    global imgnum
    cv2img = cv2.imread(img)
    # превращение фотографии в серую для работы с ней
    img_gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)

    if again == True:
    # подгрузка модели обнаружения лиц alt-2 работает получше
        face = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
    else:
        face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    # найти лица на фото. Чем меньше значения - чем больше лиц она найдет (очень много мусора)
    # faces = face.detectMultiScale(img_gray,1.001,4)
    # faces = face.detectMultiScale(img_gray,1.01,7)
    faces = face.detectMultiScale(img_gray, 1.1, 11)
    samovibor_face = False
    if type(faces) is tuple:
        print(f'[{imgnum}] Лицо на фото не было найдет, применялся коэффициент 1.1, 11. Применяем 1.01, 7')
        faces = face.detectMultiScale(img_gray,1.01,7)
        if type(faces) is tuple:
            print(f'[{imgnum}] Лицо на фото не было найдет, применялся коэффициент 1.01, 7. Применяем 1.001, 4')
            faces = face.detectMultiScale(img_gray,1.001,4)
        # print(f'[{imgnum}] Было найдено {len(faces)} лица...')
        samovibor_face = True
    print(f'[{imgnum}] Лиц найдено:{len(faces)}')  # выводит кол-во лиц
    print(f'[{imgnum}] Координаты найденных лиц: {faces}')  # выводит координаты (x,y,w,h)

    i=0
    # для каждого найденного лица
    for (x, y, w, h) in faces:
        if x < 50: x = 50  # если x находится близко к краю, то сделать его подальше
        if y < 50: y = 50  # если y находится близко к краю, то сделать его подальше
        face = cv2img[y - 50:y + h + 50, x - 50:x + w + 50]
        # print(type(x), type(y), type(w), (type(h)))
        # cv2.imshow('ree', face)  # показать обрезанное лицо
        # cv2.waitKey()  # не закрывать его
        # face_name = f'{img}({i}).png'
        # i += 1
        # cv2.imwrite(face_name, face) # временно
        if recognize_face(face):  # если вырезанное лицо совпадает с моим
            print(f'[{imgnum}] Лицо принадлежит:  Vitya')
            # cv2.imshow('ree', cv2img)  # показать обрезанное лицо
            # cv2.imshow('face', face)  # показать обрезанное лицо
            # cv2.waitKey()
            eyes = find_eyes(face)
            print('eyes in def detect_faces:', eyes)
            print(f"Середина глаз: x:{eyes['x1']+x-50},y:{eyes['y1']+y-50}, x:{eyes['x2']+x-50},y:{eyes['y2']+y-50}")
            eyes.update({'x1':eyes['x1']+x-50,'y1':eyes['y1']+y-50,'x2':eyes['x2']+x-50,'y2':eyes['y2']+y-50})
            # cv2.circle(cv2img, center=(eyes[0][0]+x-50,eyes[0][1]+y-50), radius=5, color=[255, 255, 0])
            # cv2.circle(cv2img, center=(eyes[1][0]+x-50,eyes[1][1]+y-50), radius=5, color=[255, 255, 0])
            pilimg = centre_eyes(eyes, img)

            # cv2img = cv2.cvtColor(cv2img, cv2.COLOR_RGB2BGR)
            break
        else:
            print('not vitya')
            if h == faces[-1][-1]:
                print('витя не был найден')
                global vitya
                vitya = False
                i = 0
                while not vitya:
                    for (x, y, w, h) in faces:
                        i+=1
                        if i>len(faces):
                            if again:
                                global canvas, root3, face_centre
                                face_centre=[]
                                fullpic = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)
                                image_pil = Image.fromarray(fullpic)
                                root3 = tk.Tk()
                                root3.title(f'[{imgnum}] Нажмите ЛКМ на лицо.')
                                root3.geometry('600x600')
                                canvas3 = tk.Canvas(root3, bg='white', height=1920, width=1080)
                                canvas3.pack()
                                canvas3.bind("<Button-1>", on_click4)
                                # test = PhotoImage(cv2img)
                                test = ImageTk.PhotoImage(image_pil)
                                canvas3.create_image(0, 0, anchor=tk.NW, image=test)
                                root3.mainloop()
                                face = cv2img[face_centre[1] - 300:face_centre[1] + 300, face_centre[0] - 300:face_centre[0] + 300]
                                eyes = find_eyes(face)
                                print('eyes in def detect_faces:', eyes)
                                print(
                                    f"Середина глаз: x:{eyes['x1'] + x - 300},y:{eyes['y1'] + y - 300}, x:{eyes['x2'] + x - 300},y:{eyes['y2'] + y - 300}")
                                eyes.update(
                                    {'x1': eyes['x1'] + face_centre[0] - 300, 'y1': eyes['y1'] + face_centre[1] - 300, 'x2': eyes['x2'] + face_centre[0] - 300,
                                     'y2': eyes['y2'] + face_centre[1] - 300})
                                pilimg = centre_eyes(eyes, img)
                                again = False
                                break
                                return True
                            return False
                        print(x,y,w,h)
                        if x < 50: x = 50  # если x находится близко к краю, то сделать его подальше
                        if y < 50: y = 50  # если y находится близко к краю, то сделать его подальше
                        print(x, y, w, h)
                        print(type(x),type(y),type(w),(type(h)))
                        face = cv2img[y - 50:y + h + 50, x - 50:x + w + 50]
                        global canvas2, root2
                        face2 = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        image_pil = Image.fromarray(face2)
                        root2 = tk.Tk()
                        root2.title(f'[{imgnum}] Нажмите ЛКМ, если это ваше лицо, ПКМ, если нет.')
                        root2.geometry('600x600')
                        canvas2 = tk.Canvas(root2, bg='white', height=600, width=600)
                        canvas2.pack()
                        canvas2.bind("<Button-1>", on_click2)
                        canvas2.bind("<Button-3>", on_click3)
                        # test = PhotoImage(cv2img)
                        test = ImageTk.PhotoImage(image_pil)
                        canvas2.create_image(0, 0, anchor=tk.NW, image=test)
                        root2.mainloop()
                        if vitya:
                            eyes = find_eyes(face)
                            print('eyes in def detect_faces:', eyes)
                            print(
                                f"Середина глаз: x:{eyes['x1'] + x - 50},y:{eyes['y1'] + y - 50}, x:{eyes['x2'] + x - 50},y:{eyes['y2'] + y - 50}")
                            eyes.update(
                                {'x1': eyes['x1'] + x - 50, 'y1': eyes['y1'] + y - 50, 'x2': eyes['x2'] + x - 50,
                                 'y2': eyes['y2'] + y - 50})
                            pilimg = centre_eyes(eyes, img)
                            break

    return True
    # cv2.imshow('rezult to save', cv2img)  # показать обрезанное лицо
    # cv2.waitKey()  # не закрывать его



def recognize_face(face):
    face2=cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # подгрузить датасет моих лиц
    data = pickle.loads(open('my_encoding.pickle', 'rb').read())
    # print('face2_img:', face2_img)
    # print('face:', face)
    # print('face2_img type:', type(face2_img))
    # print('face type:', type(face))
    try:
        face_enc = face_recognition.face_encodings(face2)[0]
    except:
        print('Кажется это не лицо')
        return False
    # print('face2_enc:', face2_enc)
    # print('face_enc:', face_enc)
    result = face_recognition.compare_faces(data['encoding'], face_enc)
    return result[0]


def on_click(event):
    global eyes_centre, canvas, root
    eyes_centre.append((event.x, event.y))
    print(f'[{imgnum}] Был добавлен глаз по координатам x:{event.x} y:{event.y}')
    print(f'[{imgnum}] Список глаз: {eyes_centre}')
    if len(eyes_centre) == 2:
        root.destroy()


def find_eyes(cv2img):
    global imgnum
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
    # eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye_tree_eyeglasses.xml')
    img_gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(img_gray, 1.1, 11)
    # eyes = eye_cascade.detectMultiScale(img_gray, 1.01, 7)
    # print('eyes1:', eyes)
    # print('eyes1 type:', type(eyes))
    samovibor_eyes = False
    if type(eyes) is tuple:
        print(f'[{imgnum}] Глаз с коэффициентами 1.1, 11 не было найдено. Применяем 1.01, 7')
        eyes = eye_cascade.detectMultiScale(img_gray,1.01,7)
        if type(eyes) is tuple:
            print(f'[{imgnum}] Глаз с коэффициентами 1.01, 7 не было найдено. Применяем 1.001, 4')
            eyes = eye_cascade.detectMultiScale(img_gray, 1.001, 4)
        print(f'[{imgnum}] Было найдено {len(eyes)} глаз. Придется вручную выбрать координаты зрачков')
        samovibor_eyes = True
    # print('eyes2:', eyes)
    # print('eyes2 type:', type(eyes))
    global eyes_centre
    eyes_centre=[]
    # print('eyes3:',eyes)
    for (x, y, w, h) in eyes:
        cv2.rectangle(cv2img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        eyecentre = (int(x + w / 2), int(y + h / 2))
        print(f'[{imgnum}] Середина глаза найденного: {eyecentre}')
        eyes_centre.append(eyecentre)
        cv2.circle(cv2img, center=eyecentre, radius=5, color=[0, 255, 0])
    # cv2.imshow('glaza', cv2img)  # показать обрезанное лицо c маской глаз
    # cv2.waitKey()  # не закрывать его
    if len(eyes_centre) != 2 or samovibor_eyes:
        eyes_centre = []
        global canvas, root
        image = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        root = tk.Tk()
        root.title(f'[{imgnum}] Сделайте по клику на центре каждого зрачка')
        root.geometry('600x600')
        canvas = tk.Canvas(root, bg='white', height=600, width=600)
        canvas.pack()
        canvas.bind("<Button-1>", on_click)
        # test = PhotoImage(cv2img)
        test=ImageTk.PhotoImage(image_pil)
        canvas.create_image(0, 0, anchor=tk.NW, image=test)
        root.mainloop()
    # print(f'[{imgnum}] Координаты глаз:{eyes_centre}')

    if eyes_centre[0][0] > eyes_centre[1][0]:
        eyes_dict = {'x1':eyes_centre[1][0], 'y1':eyes_centre[1][1],'x2':eyes_centre[0][0], 'y2':eyes_centre[0][1]}
    else:
        eyes_dict = {'x1':eyes_centre[0][0], 'y1':eyes_centre[0][1],'x2':eyes_centre[1][0], 'y2':eyes_centre[1][1]}
    return eyes_dict


def centre_eyes(eyes,img):
    global stock_eyes
    from math import tan, pi, atan,sqrt

    x1=eyes['x1']
    y1=eyes['y1']
    x2=eyes['x2']
    y2=eyes['y2']
    ygol=atan((y2-y1)/(x2-x1))
    tograd = (180/pi)

    # (h, w) = img.shape[:2]
    # center = (x1, y1)
    # M = cv2.getRotationMatrix2D(center, ygol*tograd, 1 )
    # img = cv2.warpAffine(img, M, (w, h))

    # cv2.imshow('re2e', img)  # показать обрезанное лицо
    # cv2.waitKey()  # не закрывать его


    im1 = Image.open(img)
    canvas = Image.new("RGBA", (1920,1080))
    # Вставляем масштабированное изображение на новое изображение с учетом сдвига
    # canvas.paste(im1,(int(shift_x), int(shift_y)))
    im2 = Image.open(r'C:\Users\Pekarnya\Downloads\unnamed.png')

    im1= im1.rotate(angle=ygol*tograd,center = (x1, y1))
    distance_between_eyes = ((x2 - x1)**2)**0.5

    stock_distance = ((stock_eyes['x1'] - stock_eyes['x2'])**2)**0.5
    scale = stock_distance/distance_between_eyes
    # print('stock dist:', stock_distance)
    # print('dist:', distance_between_eyes)
    # print('scale:', scale)
    shift_x = stock_eyes['x1'] - x1*scale
    shift_y = stock_eyes['y1'] - y1*scale
    # print('shift_x:', shift_x)
    # print('shift_y:', shift_y)
    im1 = im1.resize((int(im1.width*scale), int(im1.height * scale)))
    canvas.paste(im1, (int(shift_x), int(shift_y)))
    # scaled_image = original_image.resize((int(original_image.width * scale_factor), int(original_image.height * scale_factor)))
    # im1 = im1.resize((int(im1.width*scale), int(im1.height * scale)))

    # im1 = im1.transform(im1.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y))
    # im1.show()
    # canvas.show()
    global imgnum
    os.makedirs('08-19', exist_ok=True)
    canvas.save(fr"08-19\{imgnum:02}.png")

    # print('ygol:', ygol * tograd)
    # print('ygol 45:',tan(1/1))
    # print('ygol 45:',atan(1/1)* tograd)

    return img


stock_eyes = {'x1':836,'y1':470,'x2':1032,'y2':470}
i=0

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
    global imgnum
    imgnum = 1
    # загрузка фотографии для сравнения

    # img = 'stock.png'
    # img = '93.png'
    # img = '110.jpg'
    # img = '01.png'
    img = '22.png'

    # print(train_model(img_gray))

    # if not detect_faces(img, False):
    #     detect_faces(img, True)

    # recognize_face(img)



    directory = 'empty/'

    # Перебираем файлы в директории
    for filename in os.listdir(directory):
        # Полный путь к файлу
        filepath = os.path.join(directory, filename)
        # Проверяем, является ли объект файлом
        if os.path.isfile(filepath):
            # Делаем что-то с файлом
            print(f"[{imgnum}] Работаем с файлом '{filename}'")
            if not detect_faces(filepath, False):
                detect_faces(filepath, True)
            imgnum+=1

if __name__ == '__main__':
    main()
