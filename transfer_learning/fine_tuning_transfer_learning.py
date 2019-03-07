from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
from keras.applications import VGG16
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# dimensions of our images.
img_width, img_height = 256, 256

train_data_dir = 'jaffe2/train'
validation_data_dir = 'jaffe2/validation'
nb_train_samples = 400
nb_validation_samples = 100
epochs = 6
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
vgg_conv = VGG16(weights='imagenet',
                include_top=False,
                input_shape=(img_width, img_height, 3))

model = Sequential()
model.add(vgg_conv)
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid')) #sigmoid instead of softmax because of two classes


#make the outer layers available for fine tuning
vgg_conv.trainable = True
set_trainable = False
for layer in vgg_conv.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
vgg_conv.summary()
model.summary()
optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy',		#does not suffer from slow convergence when using sigmoid
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
model.save_weights('pretrained_fine_tuned_model.h5')
