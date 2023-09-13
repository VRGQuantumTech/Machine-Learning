# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 18:03:39 2023

@author: Victor Rollano

First steps in Machine Learning with Tensorflow and Keras
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

#%%
""" FIRST EXERCISE:
        Define a neural network (a model) with keras
        Define a set of training data and train the model
        Make a prediction with the trained model
"""

def train_model(xs,ys, epochs = 500):
    
    """
        This function creates a Sequential neural network of one neuron and
        trains it with the provided data (xs, ys)
    """
    model = keras.Sequential()
        	#creating a sequential neural network in which the layers are
        #organized sequentially. With the method model.add we can
        #introduce a layer in the model.
        
    model.add(keras.layers.Dense(units = 1, input_shape = [1]))
        #units parameter defines the number of neurons in the layer.
        #input_shape defines the dimensions of the input data.
        
    model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
        #loss function evaluates the cost of a prediction against the actual
        #value during training.
        #optimizer function creates a new prediction considering the loss
        #value of the previous prediction.
    
    model.fit(xs, ys, epochs = epochs)
        #fit trains the model with the given data.

    return model

def _linear(xs):
    
    """ This function creates the traning data I'll use. There is an input
        data xs (we could call it labels) and there is an output data ys which
        is related to xs in some manner (in this case with a linear relation
        2x - 1 but it could be other relation).
    """
    
    return 2*xs - 1

#%%

xs = np.linspace(-1., 9., dtype = float)
ys = _linear(xs)

"""
    Now we have a list with different input paramenters (the labels) and an
    array of data related to the input parameters in someway (in this case,
    2x - 1, but the neural network does not know that). In this case, each
    element in xs is one value of the same lavel so there is only one type
    of label in this dataset:
        
        xs = [label, label, label, label, ....]
    
    The numbers in ys are related to the specific label in the same position:
        
        ys = [data, data, data, data, ....]
    
    Both lists should have the same dimension since each data number in ys
    should have a label.
    
    Now we have the labels xs and the data ys, let's create and train the
    neural network with the function train_model():
"""

model = train_model(xs,ys)

print(model.predict([10.]))

#%%
"""
    SECOND EXERCISE: IMAGE RECOGNITION
    
"""

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if (logs.get('accuracy') > 0.9):
            print('\nAccuracty is enough so cancelling training!')
            self.model.stop_training = True
            

callbacks = myCallback()
fmnist = keras.datasets.fashion_mnist
(train_images, train_labels) , (test_images, test_labels) = fmnist.load_data()

print('Maximum value in data is: ' + str(train_images.max()))

train_images = train_images/train_images.max()
test_images = test_images/test_images.max()

data_shape = train_images.shape

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape = (data_shape[1], data_shape[2])))
model.add(keras.layers.Dense(units = 256, activation = tf.nn.relu))
model.add(keras.layers.Dense(units = 256, activation = tf.nn.relu))
model.add(keras.layers.Dense(units = 10, activation = tf.nn.softmax))

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs = 50, callbacks = [callbacks])

#%%
evaluation = model.evaluate(test_images, test_labels)
print(evaluation)
classifications = model.predict(test_images)
print(classifications[0])




