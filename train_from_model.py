from data_preparation import data_pre

data_train,data_validation,data_test=data_pre()

from keras.models import load_model

model=load_model('mstar10_no_bn.h5')
print(model.summary())
model.fit_generator(generator=data_train,epochs=32,
    validation_data=data_validation)
model.save('mstar10_1.h5')