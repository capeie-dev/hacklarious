import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.common.set_image_dim_ordering('th')
from matplotlib.image import imread
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 



# ## Load the data from the folder of .npy files
# Load the numpy array images onto the memory
arm = np.load('data/arm.npy')
apple = np.load('data/apple.npy')
cat = np.load('data/cat.npy')
car = np.load('data/car.npy')
dog = np.load('data/dog.npy')
horse = np.load('data/horse.npy')
face = np.load('data/face.npy')
banana = np.load('data/banana.npy')
bus = np.load('data/bus.npy')
bat = np.load('data/bat.npy')
lit = {
    0:'arm',1:'apple',2:'cat',3:'car',4:'dog',5:'horse',6:'face',7:'banana',8:'bus',9:'bat'
}
# print number of images in dataset and numpy array size of each image


# add a column with labels, 0=cat, 1=sheep, 2=cat, 3=car 
arm = np.c_[arm, np.zeros(len(arm))]
apple = np.c_[apple, np.ones(len(apple))]
cat = np.c_[cat, np.full(len(cat),2)]
car = np.c_[car, np.full(len(car),3)]
dog = np.c_[dog, np.full(len(dog),4)]
horse = np.c_[horse, np.full(len(horse),5)]
face = np.c_[face, np.full(len(face),6)]
banana = np.c_[banana, np.full(len(banana),7)]
bus = np.c_[bus, np.full(len(bus),8)]
bat = np.c_[bat, np.full(len(bat),9)]
#Function to plot 28x28 pixel drawings that are stored in a numpy array.
#Specify how many rows and cols of pictures to display (default 4x5).  
#If the array contains less images than subplots selected, surplus subplots remain empty.
with tf.device("/gpu:0"):

# merge the arm, apple, cat and car arrays, and split the features (X) and labels (y). Convert to float32 to save some memory.
    X = np.concatenate((arm[:5000,:-1], apple[:5000,:-1], cat[:5000,:-1], car[:5000,:-1]), dog[:5000,:-1], horse[:5000,:-1], face[:5000,:-1], banana[:5000,:-1], bus[:5000,:-1], bat[:5000,:-1], axis=0).astype('float32')
    y = np.concatenate((arm[:5000,-1], apple[:5000,-1], cat[:5000,-1], car[:5000,-1]), dog[:5000,:-1], horse[:5000,:-1], face[:5000,:-1], banana[:5000,:-1], bus[:5000,:-1], bat[:5000,:-1], axis=0).astype('float32') # the last column

    # train/test split (divide by 255 to obtain normalized values between 0 and 1)
    # I will use a 50:50 split, since I want to start by training the models on 5'000 samples and thus have plenty of samples to spare for testing.
    X_train, X_test, y_train, y_test = train_test_split(X/255.,y,test_size=0.5,random_state=0)


    # ## CNN part
    # one hot encode outputs
    y_train_cnn = np_utils.to_categorical(y_train)
    y_test_cnn = np_utils.to_categorical(y_test)
    num_classes = y_test_cnn.shape[1]

    # reshape to be [samples][pixels][width][height]
    X_train_cnn = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test_cnn = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# define the CNN model
def cnn_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu',data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

with tf.device("/gpu:0"):
    np.random.seed(0)
    # build the model
    model_cnn = cnn_model()
    # Fit the model
    model_cnn.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=15, batch_size=200)
    # Final evaluation of the model
    scores = model_cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)

    print('Final CNN accuracy: ', scores[1])
    # Saving the model prediction
    y_pred_cnn = model_cnn.predict_classes(X_test_cnn, verbose=0)
    model_cnn.save('my_model.h5')
    image = cv2.imread('load.jpg')
    image = cv2.resize(image, (28, 28))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape((-1, 28, 28,1)) 
    image = np.moveaxis(image, -1, 0)
    digit = model_cnn.predict_classes(image)
    print(lit[digit[0]])
    # Finding the accuracy score
    acc_cnn = accuracy_score(y_test, y_pred_cnn)

    print ('CNN accuracy: ',acc_cnn)