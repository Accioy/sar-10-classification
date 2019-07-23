import os
from keras.preprocessing.image import ImageDataGenerator

def data_pre():
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
    data_test=test_generator.flow_from_directory(directory=os.path.join(root_path,'Mstar_dataset','test'),
    target_size=[128,128],color_mode='grayscale',class_mode='categorical',
    batch_size=32,shuffle=True
)
    return data_train,data_validation,data_test

