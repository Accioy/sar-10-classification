import keras
from data_preparation import data_pre


from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, Dense, BatchNormalization, Dropout, MaxPool2D, Flatten
from keras.metrics import categorical_accuracy

from keras import optimizers


vgg16_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', 
    input_shape=[128,128,3], pooling='max')

x=vgg16_model.output
x=Dense(4096,activation = 'relu')(x)
x=Dropout(0.5)(x)
x=Dense(4096,activation = 'relu')(x)
x=Dropout(0.5)(x)
predict=Dense(10, activation = 'softmax')(x)

model=Model(inputs=vgg16_model.input, outputs=predict)
adam=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['categorical_accuracy'])

print(model.summary())


data_train,data_validation,data_test=data_pre()

model.fit_generator(generator=data_train,epochs=32,
    validation_data=data_validation)
model.save('mstar10-vgg16.h5')



