import os
import time
import keras
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import keras.utils as image
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
from keras import backend as BK
import tensorflow as tf

# Define the model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))#Using (448,448) will run out of allocated memory
model.add(MaxPooling2D())
# model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D())
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(13, activation='softmax'))

#memory and parameter using status
model.summary()
with tf.device('/cpu:0'):
    model.build((224,224,3))
    model.summary()
    tf.profiler.experimental.Profile(model, tf.profiler.experimental.ProfilerOptions(host_tracer_level=1))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Preprocess image datas
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
# path_train = '/home/u/Pictures/ml_bg_train'
path_train = '/home/u/Pictures/ml_bg_train'
train_generator = datagen.flow_from_directory(
    path_train,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)
train_class_label = list(train_generator.class_indices.keys())
print('Train class labels: ', train_class_label)

path_validation = '/home/u/Pictures/ml_bg_validation'
valid_generator = datagen.flow_from_directory(
    path_validation,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)
valid_class_label = list(valid_generator.class_indices.keys())
print('Valid class labels: ', valid_class_label)

#Dashbord and logs
log_dir = '/home/u/ml_bg_log/' + time.strftime('%Y%m%d-%H%M%S')
tensorBoard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


#Monitor the training
# for epoch in range(10):
#     print("Epoch ", epoch+1, ":")
#     history = model.fit(train_generator,
#       shuffle=True,
#       batch_size=128,
#       epochs=1,
#       validation_data=valid_generator)
#     prob = model.predict_generator(valid_generator)
#     print("Probability Output: ", prob[:10])

#Train the model
his = model.fit(
    train_generator,
    shuffle=True,
    steps_per_epoch=len(train_generator)//32,
    epochs=20,
    validation_data=valid_generator,
    validation_steps=len(valid_generator)//32,
    callbacks=[tensorBoard_callback]
)

print(len(train_generator),"\n",len(valid_generator))
#Draw the
var_loss = his.history['val_loss']
plt.plot(np.arange(20)+1, var_loss, label='Originalm model')
plt.title("Effect of model capacity on validation loss\n")
plt.xlabel('Epoch #')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()


#Save model to load
model.save('/home/u/my_model')
model = keras.models.load_model('/home/u/my_model')

#Using well trained model to catatory pic
train_class_label = ['Blue_flower_by_Elena_Stravoravdi.jpg', 'Cherry_Tree_in_Lakones_by_elenastravoravdi.jpg', 'DSC2943_by_kcpru.jpg', 'Jammy-Jellyfish_WP_4096x2304_Grey.png', 'Mirror_by_Uday_Nakade.jpg', 'Optical_Fibers_in_Dark_by_Elena_Stravoravdi.jpg', 'canvas_by_roytanck.jpg', 'jj_dark_by_Hiking93.jpg', 'jj_light_by_Hiking93.jpg', 'ubuntu-default-greyscale-wallpaper.png', 'ubuntu2_by_arman1992.jpg', 'ubuntu_by_arman1992.jpg', 'warty-final-ubuntu.png']
img_path = '/home/u/Pictures/test_ml_Screenshots/Screenshot from 2023-02-21 03-48-17.png'
img = image.load_img(img_path,target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
pred_point = model.predict(x)
print(pred_point)
print('Predict image is:', train_class_label[np.argmax(pred_point)])
os.system('tensorboard --logdir '+ log_dir )


