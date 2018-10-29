"""
MNIST Classification using 2D convolutional neural networks in Keras (Tensorflow)
@author: Aqib Ansari
"""

## import

    # import keras - neural network
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

    # import matplotlib - Data Visualisation
from matplotlib.pylab import plt

    # import Dataset
from keras.datasets import mnist
     
## Preprosessing the Dataset

     # Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

    # parameters that describe the input data
img_x = x_train.shape[1]
img_y = x_train.shape[2]
input_shape = (img_x, img_y, 1)
num_classes = 10

    # visualise data
plt.imshow(x_train[0])

    # explore the dataset
print(y_train[1:10])

    # convert data into a binary form
y_train = np_utils.to_categorical(y_train, 10)

    # reshape data into a 4d tensor
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

    # change data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

    # normalise the data
x_train /= 255
x_test /= 255

## build a neural network

    # parameters of the neural network
num_classes = 10

    # type of model
model = Sequential()

    # first convolutional layer
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))

    # second convolutional layer
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

    # convert the image (2D array) into a vector
model.add(Flatten())

    # fully connected hidden layer
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))

    # fully connected output layer
model.add(Dense(num_classes, activation = 'softmax'))

## compiling the neural network
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

## training the neural network

    # parameters to train the model
epochs = 1
batch_size = 2

    #  train the neural network
model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose =1)

## prediction
score = model.evaluate(x_test, y_test, verbose = 0)
