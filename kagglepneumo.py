import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import  ImageDataGenerator


def get_arg():
    parser = argparse.ArgumentParser(description='Neural network used to predict pneumonia from lungs radiographies')
    parser.add_argument('data', nargs=3, metavar='data', type=str, help="The datas to be used (train set, test set and"
                                                                        " validation set)")
    return parser.parse_args()


def get_path(path, state='Normal'):
    full_path = False
    if state == "Normal":
        full_path = path + 'NORMAL/'
    if state == "Pneumonia":
        full_path = path + 'PNEUMONIA/'
    return full_path


def get_image(pathi):
    if pathi:
        img = Image.open(pathi)
        return img.resize((250, 250))


def get_images_name(path):
    image_list = sorted(os.listdir(path))
    return image_list


def fill_up_array(path, images, label, array):
    for image in images:
        img = get_image(path + label + '/' + image)
        array.append([img, label])


def array_image(path):
    print("      making array of images in process...")
    image_array = []
    normal_list = get_images_name(get_path(path))
    pneumonia_list = get_images_name(get_path(path, "Pneumonia"))
    fill_up_array(path, normal_list, "NORMAL", image_array)
    fill_up_array(path, pneumonia_list, "PNEUMONIA", image_array)
    random.shuffle(image_array)
    return image_array


def image_printing(image, ref):
    print("image printing")
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(ref)
    plt.show()
    print("image shown")


def reshaping(array):
    xtrain = []
    ytrain = []
    for image, label in array:
        xtrain.append(image)
        if label == "NORMAL":
            ytrain.append(0)
        else:
            ytrain.append(1)
    return np.array(xtrain), np.array(ytrain)


def get_data(path):
    datagen = ImageDataGenerator(rescale=1. / 255)
    dataset = datagen.flow_from_directory(path, target_size=(150, 150), batch_size=16,
                                                     class_mode='binary')
    return dataset


def modelp(shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation="softmax"))
    return model


def main():
    print("Program launched")
    args = get_arg()
    print("Retrieving radiography datas :")
    print("   retrieving training data")
    xtrain = get_data(args.data[0])
    print("   retrieving testing data")
    xtest = get_data(args.data[1])
    print("   retrieving validation data")
    xval = get_data(args.data[2])
    print("data retrieved")
    print(xtrain.labels)
    print(xtest.labels)
    print(xval.labels)
    #image_printing(array_train[0][0], array_train[0][1])
    print("Reshaping done")
    print("Writing model:")
    input_shape = (28, 28, 1)
    print("  categorizing done...")
    model1 = modelp(input_shape)
    model1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history1 = model1.fit(xtrain, steps_per_epoch=128, epochs=15, validation_data=xtest)
    _, acc = model1.evaluate(xtest, verbose=1)
    print('> %.3f' % (acc * 100.0))
    print(history1.history.keys())
    print("Program done")


if __name__ == '__main__':
    main()
