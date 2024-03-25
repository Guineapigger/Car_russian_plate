import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import json

model_numbers = keras.models.load_model('model_numbers.keras')
model_letters = keras.models.load_model('model_letters.keras')
model_region = keras.models.load_model('model_region.keras')

def open_img(name):
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def car_plate_find(img, haar_cascade):
    res = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h in res:
        car_plate_img = img[y + 15:y + h - 10, x + 10:x + w - 10]
        return car_plate_img


def enlarge_img(img, percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img


data_numbers = []
data_letters = []
data_region = []

file_name = rf'B932TT27.png'
img = open_img(file_name)
car_plate_coord = enlarge_img(img, 150)
gray_img = cv2.cvtColor(car_plate_coord, cv2.COLOR_RGB2GRAY)
resize_img = cv2.resize(gray_img, (255, 50))

data_letters.append(cv2.bitwise_not(cv2.resize(resize_img[:, 10:33], (28, 28)))) # в список data_letters попадают двумерный массив, представляющей цифры
data_letters.append(cv2.bitwise_not(cv2.resize(resize_img[:, 121:145], (28, 28))))
data_letters.append(cv2.bitwise_not(cv2.resize(resize_img[:, 147:171], (28, 28))))

data_numbers.append(cv2.bitwise_not(cv2.resize(resize_img[:, 43:67], (28, 28)))) # в список data_numbers попадают двумерный массив, представляющей буквы
data_numbers.append(cv2.bitwise_not(cv2.resize(resize_img[:, 69:93], (28, 28))))
data_numbers.append(cv2.bitwise_not(cv2.resize(resize_img[:, 95:119], (28, 28))))

data_region.append(cv2.bitwise_not(cv2.resize(resize_img[:, 180:203], (28, 28))))
data_region.append(cv2.bitwise_not(cv2.resize(resize_img[:, 205:225], (28, 28))))
data_region.append(cv2.bitwise_not(cv2.resize(resize_img[:, 226:], (28, 28))))

t = {0: 'A', 1: 'B', 2: 'E', 3: 'K', 4: 'M', 5: 'H', 6: 'O', 7: 'P', 8: 'C', 9: 'T', 10: 'Y', 11: 'X'}
d = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: ' '}
data_region = np.array(data_region) / 255
data_numbers = np.array(data_numbers) / 255
data_letters = np.array(data_letters) / 255

otv = []

x = np.expand_dims(data_letters[0], axis=0)
x = model_letters(x)
otv.append(t[np.argmax(x)])


x = np.expand_dims(data_numbers[0], axis=0)
x = model_numbers(x)
otv.append(str(np.argmax(x)))


x = np.expand_dims(data_numbers[1], axis=0)
x = model_numbers(x)
otv.append(str(np.argmax(x)))

x = np.expand_dims(data_numbers[2], axis=0)
x = model_numbers(x)
otv.append(str(np.argmax(x)))

x = np.expand_dims(data_letters[1], axis=0)
x = model_letters(x)
otv.append(t[np.argmax(x)])

x = np.expand_dims(data_letters[2], axis=0)
x = model_letters(x)
otv.append(t[np.argmax(x)])

x = np.expand_dims(data_region[0], axis=0)
x = model_region(x)
otv.append(d[np.argmax(x)])

x = np.expand_dims(data_region[1], axis=0)
x = model_region(x)
otv.append(d[np.argmax(x)])

x = np.expand_dims(data_region[2], axis=0)
x = model_region(x)
otv.append(d[np.argmax(x)])

print(''.join(otv))