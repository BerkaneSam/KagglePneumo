import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import os
import random


def get_arg():
    parser = argparse.ArgumentParser(description='Neural network used to predict pneumonia from lungs radiographies')
    parser.add_argument('data', nargs=3, metavar='data', type=str, help="The data to be used (train set, test set and"
                                                                        "validation set)")
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
    print("      making array of image in process...")
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
        ytrain.append(label)
    return np.array(xtrain), np.array(ytrain)


def main():
    print("Program launched")
    args = get_arg()
    print("Retrieving radiography datas :")
    print("   retrieving training data")
    array_train = array_image(args.data[0])
    print("   retrieving testing data")
    array_test = array_image(args.data[1])
    print("   retrieving validation data")
    array_val = array_image(args.data[2])
    print("data retrieved")
    image_printing(array_train[0][0], array_train[0][1])
    print("Reshaping data")
    xtrain, ytrain = reshaping(array_train)
    xtest, ytest = reshaping(array_test)
    xval, yval = reshaping(array_val)
    print("Program done")


if __name__ == '__main__':
    main()
