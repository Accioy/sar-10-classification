import os
from keras.preprocessing.image import ImageDataGenerator,load_img

from keras.models import load_model
from keras.layers import Conv2D, Dense, BatchNormalization, Dropout, MaxPool2D, Flatten
from keras.metrics import categorical_accuracy

import numpy as np
import matplotlib.pyplot as plt

root_path=os.path.abspath('./')
# test_generator=ImageDataGenerator(rescale=1./255)
# data_test=test_generator.flow_from_directory(directory=os.path.join(root_path,'Mstar_dataset','validation'),
#     target_size=[128,128],color_mode='grayscale',class_mode='categorical',
#     batch_size=32,shuffle=True
# )

model=load_model('mstar10_1.h5')
print(model.summary())
test_sample=os.path.join(root_path,'Mstar_dataset','validation','2S1','hb19381.jpeg')
test_sample_image = load_img(test_sample,grayscale=True,target_size=[128,128])
test_sample_image = np.asarray(test_sample_image)
test_sample_image=np.reshape(test_sample_image,(-1,128,128,1))
test_sample_image=test_sample_image/255.0

# result=model.predict_on_batch(test_sample_image)
# print(result)
# [[9.9999785e-01 7.1729005e-11 3.1714440e-13 7.9016854e-10 1.3168177e-09
#   2.1032763e-10 3.0747106e-11 3.4430950e-17 2.1424542e-06 2.5695775e-17]]

from keras import backend as K

Layer_names=['max_pooling2d_1','max_pooling2d_2','max_pooling2d_3','max_pooling2d_4']
Mean_activations=[]
for layer_name in Layer_names:
    L=model.get_layer(name=layer_name)
    f1 = K.function([model.layers[0].input], [L.output])
    feature=f1([test_sample_image])[0][0]
    feature=np.transpose(feature,(2,0,1))
    print(feature.shape)
    # for i in range(16):
    #     plt.subplot(4,4,i+1)
    #     plt.imshow(feature[i]) #,cmap='gray'  
    # plt.savefig(os.path.join(root_path,'vi',layer_name+'.jpg'))

    Mean_activation=[]
    for channel in feature:
        Mean_activation.append(channel.mean())
    Mean_activations.append(Mean_activation)

i=1
for activations in Mean_activations:
    values=np.array(activations)
    x=np.linspace(1,len(activations),len(activations))
    plt.stem(x,values)
    # plt.imshow(img)
    plt.savefig(os.path.join(root_path,'vi','activations',str(i)+'.jpg'))
    i+=1

