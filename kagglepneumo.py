import numpy as np
import argparse
import os
import cv2
from tensorflow.keras import layers, models


def get_arg():
    parser = argparse.ArgumentParser(description='Neural network used to predict pneumonia from lungs radiographies')
    parser.add_argument('data', nargs=3, metavar='data', type=str, help="The datas to be used (train set, test set and"
                                                                        " validation set)")
    return parser.parse_args()


def modelp():
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation="softmax"))
    return model


def get_data(path, size):
    data = []
    labels = ["PNEUMONIA", "NORMAL"]
    for label in labels:
        fullpath = os.path.join(path, label)
        class_num = labels.index(label)
        for img in os.listdir(fullpath):
            arr_img = cv2.imread(os.path.join(fullpath, img), cv2.IMREAD_GRAYSCALE)
            arr_res = cv2.resize(arr_img, (size, size))
            data.append([arr_res, class_num])
    return np.array(data)


def preprocessing_data(data):
    print("   Preprocessing ongoing...")
    x_data = []
    y_data = []
    for img, label in data:
        x_data.append(img)
        y_data.append(label)
    return x_data, y_data


def data_normalization(xdata, ydata, size):
    print("   Normalization onngoing")
    xdata = np.array(xdata) / 255
    xdata = xdata.reshape(-1, size, size, 1)
    return xdata, np.array(ydata)


def main():
    print("Program launched")
    args = get_arg()
    print("Retrieving data")
    set_train = get_data(args.data[0], 150)
    print("   training data retrieved")
    set_test = get_data(args.data[1], 150)
    print("   testing data retrieved")
    set_val = get_data(args.data[2], 150)
    print("   validation data retrieved")
    print("Preprocessing starting...")
    xtrain, ytrain = preprocessing_data(set_train)
    print("   training preprocessing over")
    xtest, ytest = preprocessing_data(set_test)
    print("   testing preprocessing over")
    xval, yval = preprocessing_data(set_val)
    print("   validation preprocessing over")
    print("Normalisation starting...")
    x_train, y_train = data_normalization(xtrain, ytrain, 150)
    print("   training normalization over")
    x_test, y_test = data_normalization(xtest, ytest, 150)
    print("   testing normalization over")
    x_val, y_val = data_normalization(xval, yval, 150)
    print("Writing model...")
    model1 = modelp()
    model1.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    print("Model learning...")
    history1 = model1.fit(x_train, y_train, batch_size=128, epochs=15, validation_data=(x_val, y_val))
    print(history1.history.keys())
    print("Model evaluation:")
    eval_model = model1.evaluate(x_test, y_test)
    print(f"Model loss : {eval_model[0]}")
    print(f"Model accuracy : {eval_model[1]*100}%")
    print("Program done")


if __name__ == '__main__':
    main()
