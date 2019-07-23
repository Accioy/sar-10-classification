from keras.models import load_model
from keras import backend as K
from keras import utils

import os
import numpy as np
import matplotlib.pyplot as plt


model=load_model('mstar10_1.h5')
root_path=os.path.abspath('./')


def normalize(x):
    """utility function to normalize a tensor.
    # Arguments
        x: An input tensor.
    # Returns
        The normalized input tensor.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def conv_filter(model, layer_name, img):
    """Get the filter of conv layer.

    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.

    Returns:
           filters.
    """
    # this is the placeholder for the input images
    input_img = model.input

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    try:
        layer_output = layer_dict[layer_name].output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    kept_filters = []
    for i in range(layer_output.shape[-1]):
        loss = K.mean(layer_output[:, :, :, i])

        # compute the gradient of the input picture with this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1
        # run gradient ascent for 20 steps
        fimg = img.copy()

        for j in range(100):
            loss_value, grads_value = iterate([fimg])
            fimg += grads_value * step

        # decode the resulting input image
        # fimg = utils.deprocess_image(fimg[0])
        fimg=fimg[0]
        kept_filters.append((fimg, loss_value))

        # sort filter result
        kept_filters.sort(key=lambda x: x[1], reverse=True)

    return np.array([f[0] for f in kept_filters])

input_img=np.random.normal(size=[1,128,128,1])

Layer_names=['max_pooling2d_1','max_pooling2d_2','max_pooling2d_3','max_pooling2d_4']
for layer_name in Layer_names:

    result=conv_filter(model,layer_name,input_img)

    for i in range(16):
        img=result[i]
        img=img[:,:,0]
        plt.subplot(4,4,i+1)
        plt.imshow(img,cmap='gray')
    plt.savefig(os.path.join(root_path,'vi','gradient_acent',layer_name+'.jpg'))
    # plt.show()