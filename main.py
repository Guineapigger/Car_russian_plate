import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import json


def open_img(name):
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def car_plate_find(img, haar_cascade):
    res = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h in res:
        car_plate_img = img[y+15:y+h-10, x+10:x+w-10]
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
answer_numbers = []
answer_letters = []
answer_region = []


path = os.walk(r'E:\forTrain\img')
haar_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')


for lst in path:
    for i in lst[2]:
        file_name = rf'E:\forTrain\img\{i}'
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


path = os.walk(r'E:\forTrain\ann')

for lst in path:
    for i in lst[2]:
        file_name = rf'E:\forTrain\ann\{i}'
        file = json.load(open(file_name))
        otv = file['description']
        if len(otv) == 8:
            otv += ' '
            otv = str(otv)

        answer_numbers.append(otv[1])
        answer_numbers.append(otv[2])
        answer_numbers.append(otv[3])

        answer_letters.append(otv[0])
        answer_letters.append(otv[4])
        answer_letters.append(otv[5])

        answer_region.append(otv[6])
        answer_region.append(otv[7])
        answer_region.append(otv[8])




for i in range(len(answer_numbers)):
    answer_numbers[i] = int(answer_numbers[i])

letters_dict = {'A': 0, 'B': 1, 'E': 2, 'K': 3, 'M': 4, 'H': 5, 'O': 6, 'P': 7, 'C': 8, 'T': 9, 'Y': 10, 'X': 11}

region_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, ' ': 10}

for i in range(len(answer_letters)):
    answer_letters[i] = letters_dict[answer_letters[i]]

for i in range(len(answer_region)):
    answer_region[i] = region_dict[answer_region[i]]

answer_numbers = np.array(answer_numbers)
data_numbers = np.array(data_numbers)

data_numbers = data_numbers / 255

answer_numbers = keras.utils.to_categorical(answer_numbers, 10)

# print(len(data_numbers))
# print(len(answer_numbers))
#
# print(len(data_letters))
# print(len(answer_letters))
#
# print(len(data_region))
# print(len(answer_region))

model_numbers = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')])
model_numbers.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model_numbers.fit(data_numbers, answer_numbers, batch_size=32, epochs=20, validation_split=0.2)

print("  ")

answer_letters = np.array(answer_letters)
data_letters = np.array(data_letters)

data_letters = data_letters / 255

answer_letters = keras.utils.to_categorical(answer_letters, 12)

model_letters = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(12, activation='softmax')])
model_letters.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model_letters.fit(data_letters, answer_letters, batch_size=32, epochs=20, validation_split=0.2)

print("  ")

data_region = np.array(data_region)
answer_region = np.array(answer_region)

data_region = data_region / 255

answer_region = keras.utils.to_categorical(answer_region, 11)

model_region = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(11, activation='softmax')])
model_region.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model_region.fit(data_region, answer_region, batch_size=32, epochs=20, validation_split=0.2)

print("  ")


model_region.save('model_region.keras')
model_numbers.save('model_numbers.keras')
model_letters.save('model_letters.keras')


data_numbers = []
data_letters = []
data_region = []

file_name = rf'21_5_2014_19_1_8_895_0.png'
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