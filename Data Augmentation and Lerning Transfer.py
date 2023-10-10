# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 21:44:23 2023

@author: vrgns
"""

import os
import zipfile
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#%% Preparing data with ImageDataGenerator

zip_ref = zipfile.ZipFile('cats_and_dogs_filtered.zip', 'r')
zip_ref.extractall()
zip_ref.close()

base_dir = 'cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')



train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size = (150,150),
                    batch_size = 20,
                    class_mode = 'binary')

validation_datagen = ImageDataGenerator(rescale = 1./255.)

validation_generator = validation_datagen.flow_from_directory(
                    validation_dir,
                    target_size = (150,150),
                    batch_size = 20,
                    class_mode = 'binary')

#%% Learning Transfer and Neural Network definition

local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150,150,3),
                                include_top = False,
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

# with this loop layers in the pretained model are locked, they won't be
# trainable.
for layer in pre_trained_model.layers:
    layer.trainable = False
    
pre_trained_model.summary()

# Taiking layer named mixed7
last_layer = pre_trained_model.get_layer('mixed7') 

# Obtaining output from layer mixed7
last_output = last_layer.output

# Now the new model take the output from mixed7 layer from inception model
x = Flatten()(last_output)
x = Dense(1024, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer = RMSprop(learning_rate=0.0001),
              loss = 'binary_crossentropy',
              metrics = ['acc'])

model.summary()

#%%

history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 20,
            epochs = 30,
            validation_steps = 5,
            verbose = 2)

#%%
from matplotlib import pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.figure()
plt.plot(history.epoch, acc, color = 'red', label = 'Training Accuracy')
plt.plot(history.epoch, val_acc, color = 'royalblue', label = 'Validation Accuracy')
plt.tick_params(direction = 'in')
plt.title('Dense layers', fontweight='bold' )
plt.xlabel('# Epochs' , fontsize = 14, fontweight='bold')
plt.ylabel('Accuracy', fontsize = 14, fontweight='bold')
plt.legend()
plt.show()
plt.close()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.plot(history.epoch, loss, color = 'red', label = 'Training Loss')
plt.plot(history.epoch, val_acc, color = 'royalblue', label = 'Validation Loss')
plt.tick_params(direction = 'in')
plt.title('Dense layers', fontweight='bold' )
plt.xlabel('# Epochs' , fontsize = 14, fontweight='bold')
plt.ylabel('Loss', fontsize = 14, fontweight='bold')
plt.legend()
plt.show()
plt.close()

#%% EXPLORING DROPOUTS

import tensorflow as tf
from tensorflow.keras.layers import Dropout

# introducing a callback just to practice
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') is not None and logs.get('acc')> 0.985):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True

x = Flatten()(last_output)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.2)(x) #dropping 20% of the neurons by not connecting them
x = Dense(1, activation = 'sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer = RMSprop(lr=0.0001),
              loss = 'binary_crossentropy',
              metrics = ['acc'])

callbacks = myCallback()
history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 20,
            epochs = 30,
            validation_steps = 5,
            verbose = 2,
            callbacks = callbacks)

#%%

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.figure()
plt.plot(history.epoch, acc, color = 'red', label = 'Training Accuracy')
plt.plot(history.epoch, val_acc, color = 'royalblue', label = 'Validation Accuracy')
plt.tick_params(direction = 'in')
plt.title('With Dropout layer 20%', fontweight='bold' )
plt.xlabel('# Epochs' , fontsize = 14, fontweight='bold')
plt.ylabel('Accuracy', fontsize = 14, fontweight='bold')
plt.legend()
plt.show()
plt.close()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.plot(history.epoch, loss, color = 'red', label = 'Training Loss')
plt.plot(history.epoch, val_acc, color = 'royalblue', label = 'Validation Loss')
plt.tick_params(direction = 'in')
plt.title('With Dropout layer 20%', fontweight='bold' )
plt.xlabel('# Epochs' , fontsize = 14, fontweight='bold')
plt.ylabel('Loss', fontsize = 14, fontweight='bold')
plt.legend()
plt.show()
plt.close()


