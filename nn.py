import os
from keras.preprocessing.image import ImageDataGenerator

# data preparation

root_path=os.path.abspath('./')


train_generator=ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='constant',
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255
)

test_generator=ImageDataGenerator(rescale=1./255)


data_train=train_generator.flow_from_directory(directory=os.path.join(root_path,'Mstar_dataset','train'),
    target_size=[128,128],color_mode='grayscale',class_mode='categorical',
    batch_size=32,shuffle=True
)

data_validation=test_generator.flow_from_directory(directory=os.path.join(root_path,'Mstar_dataset','validation'),
    target_size=[128,128],color_mode='grayscale',class_mode='categorical',
    batch_size=32,shuffle=True
)



# for i in range(20):
#     gen_train.next()

from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, BatchNormalization, Dropout, MaxPool2D, Flatten
from keras.metrics import categorical_accuracy

model=Sequential()
model.add(Conv2D(input_shape=[128,128,1], filters=16, kernel_size=[3,3],activation='relu'))
# model.add(BatchNormalization(momentum=0.9))
model.add(MaxPool2D([2,2]))
model.add(Conv2D(filters=32, kernel_size=[3,3],padding='same',activation='relu'))
# model.add(BatchNormalization(momentum=0.9))
model.add(MaxPool2D([2,2]))
model.add(Conv2D(filters=64, kernel_size=[3,3],padding='same',activation='relu'))
# model.add(BatchNormalization(momentum=0.9))
model.add(MaxPool2D([2,2]))
model.add(Conv2D(filters=128, kernel_size=[3,3],padding='same',activation='relu'))
# model.add(BatchNormalization(momentum=0.9))
model.add(MaxPool2D([2,2]))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_accuracy'])
print(model.summary())

model.fit_generator(generator=data_train,epochs=100,
    validation_data=data_validation)
model.save('mstar10.h5')
