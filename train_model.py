import build_model_1                                        #     HERE CHANGING MODEL ##################################
import os
import h5py

import keras  # Keras 2.1.2 and TF-GPU 1.9.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import time

NAME = f'phase1-{int(time.time())}'




def mkdir_exists(dir):
    if os.path.exists(dir):
        return
    os.mkdir(dir)


def data_reader():
    with h5py.File(''.join(['dataset-v1.h5']), 'r') as hf:
        X_train = hf['X_train'].value
        y_train = hf['y_train'].value
        X_val = hf['X_val'].value
        y_val = hf['y_val'].value
    return X_train, y_train, X_val, y_val


def train():
    model = build_model_1.init_model()                    #     HERE CHANGING MODEL ##################################

    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    X_train, y_train, X_val, y_val = data_reader()

    model.fit(X_train, y_train, nb_epoch=1000, batch_size=len(X_train))
    score = model.evaluate(X_val, y_val, batch_size=len(X_val))

    t = time.time()

    print("Accuracy: ", score)
    print("Training duration: ", round(time.time()-t, 2))
    print("Model trained")

    keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=len(X_train), write_graph=True, write_grads=True, write_images=False, embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata=None)
    model.save(f"models/January_18---{NAME}")




    # print('\n\t\t X_train')
    # print(X_train)          #   [[], ..., []]
    # print(type(X_train))    #   <class 'numpy.ndarray'>
    # print(len(X_train))     #   712
    # print(X_train.shape)    #   (712, 7)
    # print('\n')
    # print(y_train)          #   [[], ..., []]
    # print(type(y_train))    #   <class 'numpy.ndarray'>
    # print(len(y_train))     #   712
    # print(y_train.shape)    #   (712, 1)
    # print('\n')
    # print(X_val)            #   [[], ..., []]
    # print(type(X_val))      #   <class 'numpy.ndarray'>
    # print(len(X_val))       #   179
    # print(X_val.shape)      #   (179, 7)
    # print('\n')
    # print(y_val)            #   [[], ..., []]
    # print(type(y_val))      #   <class 'numpy.ndarray'>
    # print(len(y_val))       #   179
    # print(y_val)            #   (179, 1)
    # print('\n')
    # print('\n')

    # mkdir_exists("weights")

    # training & validation


train()
