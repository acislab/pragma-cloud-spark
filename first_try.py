
# coding: utf-8

# ### Giving Keras a try, this code is based on the example from the lead Keras developer [here](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

# In[1]:


# Limit to CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
print("IMPORTS")

# The images have already been split into subdirectories for training/validation/testing. 70% in train, 20% in validation, 10% in test

# In[2]:

width, height = 256, 256


# Epochs and batch sizes can be messed around with and can make relatively big difference in the model

# In[3]:

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 280
nb_validation_samples = 80
epochs = 25
batch_size = 16


# In[4]:

if K.image_data_format() == 'channels_first':
    input_shape = (3, width, height)
else:
    input_shape = (width, height, 3)


# In[5]:

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[18]:

train_datagen = ImageDataGenerator(
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True
)


# In[19]:

validation_datagen = ImageDataGenerator()


# In[20]:

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='binary')


# In[21]:

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='binary')


# This is where the magic happens. As you can see from the accuracy here, it is far from magic, i.e. this model doesn't work. But it's a start!

# In[22]:

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


# In[23]:

#model.save_weights('first_try.h5')
print(model.to_json())

