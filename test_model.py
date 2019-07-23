import os
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, BatchNormalization, Dropout, MaxPool2D, Flatten
from keras.metrics import categorical_accuracy

root_path=os.path.abspath('./')
test_generator=ImageDataGenerator(rescale=1./255)
data_test=test_generator.flow_from_directory(directory=os.path.join(root_path,'Mstar_dataset','validation'),
    target_size=[128,128],color_mode='grayscale',class_mode='categorical',
    batch_size=32,shuffle=True
)
print(data_test.class_indices)
model=load_model('mstar10_1.h5')
evaluation=model.evaluate_generator(data_test)
print(evaluation)