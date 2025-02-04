#!/usr/bin/env python3
# mnist_cifar_models.py
##
# Model definition for MNIST and CIFAR
##
# Copyright (C) 2018, Huan Zhang <huan@huan-zhang.com> and contributors
##
# This program is licenced under the BSD 2-Clause License,
# contained in the LICENCE file in this directory.
# See CREDITS for a list of contributors.
##

import numpy as np
import os
import json
import argparse
import urllib.request

#import tensorflow as tf
import tensorflow.compat.v1 as tf
# use tf.keras or keras
use_tf_keras = False

# due to incompatibility between keras and tf.keras, we will try to import both


def get_model_meta(filename):
    print("Loading model " + filename)
    global use_tf_keras
    global Sequential, Dense, Dropout, Activation, Flatten, Lambda, Conv2D, MaxPooling2D, LeakyReLU, regularizers, K
    try:
        from keras.models import load_model as load_model_keras
        ret = get_model_meta_real(filename, load_model_keras)
        # model is successfully loaded. Import layers from keras
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
        from keras.layers import Conv2D, MaxPooling2D
        from keras.layers import LeakyReLU
        from keras import regularizers
        from keras import backend as K
        print("Model imported using keras")
    except (KeyboardInterrupt, SystemExit, SyntaxError, NameError, IndentationError):
        raise
    except:
        print("Failed to load model with keras. Trying tf.keras...")
        use_tf_keras = True
        from tensorflow.keras.models import load_model as load_model_tf
        ret = get_model_meta_real(filename, load_model_tf)
        # model is successfully loaded. Import layers from tensorflow.keras
        #since the code is written in tf 1.x, we need to use tf.compat.v1, so all "tensorflow" is changed to tensorflow.compat.v1
        #import tensorflow.compat.v1 as tf
        from tensorflow.compat.v1.keras.models import Sequential
        from tensorflow.compat.v1.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
        from tensorflow.compat.v1.keras.layers import Conv2D, MaxPooling2D
        from tensorflow.compat.v1.keras.layers import LeakyReLU
        from tensorflow.compat.v1.keras import regularizers
        from tensorflow.compat.v1.keras import backend as K
        print("Model imported using tensorflow.keras")
    # put imported functions in global
    Sequential, Dense, Dropout, Activation, Flatten, Lambda, Conv2D, MaxPooling2D, LeakyReLU, regularizers, K = \
        Sequential, Dense, Dropout, Activation, Flatten, Lambda, Conv2D, MaxPooling2D, LeakyReLU, regularizers, K
    return ret


def get_model_meta_real(filename, model_loader):
    model = model_loader(filename, custom_objects={
                         "fn": lambda y_true, y_pred: y_pred, "tf": tf})
    json_string = model.to_json()
    model_meta = json.loads(json_string)
    weight_dims = []
    activations = set()
    activation_param = None
    input_dim = []
    # print(model_meta)
    try:
        # for keras
        model_layers = model_meta['config']['layers']
    except (KeyError, TypeError):
        # for tensorflow.keras
        model_layers = model_meta['config']
    for i, layer in enumerate(model_layers):
        # if i ==0 and layer['class_name'] == "Flatten":
        if i == 0 or layer['class_name'] == "Flatten":

            input_dim = layer['config']['batch_input_shape']
        if layer['class_name'] == "Dense":
            units = layer['config']['units']
            weight_dims.append(units)
            activation = layer['config']['activation']
            if activation != 'linear':
                activations.add(activation)
        elif layer['class_name'] == "Activation":
            activation = layer['config']['activation']
            activations.add(activation)
        elif layer['class_name'] == "LeakyReLU":
            activation_param = layer['config']['alpha']
            activations.add("leaky")
        elif layer['class_name'] == "Lambda":
            if "arctan" in layer['config']["name"]:
                activation = "arctan"
                activations.add("arctan")
    assert len(activations) == 1, "only one activation is supported," + \
               str(activations)
    return weight_dims, list(activations)[0], activation_param, input_dim


class NLayerModel:
    def __init__(self, params, restore=None, session=None, use_softmax=False, image_size=28, image_channel=1, activation='relu', activation_param=0.3, l2_reg=0.0, dropout_rate=0.0):

        global Sequential, Dense, Dropout, Activation, Flatten, Lambda, Conv2D, MaxPooling2D, LeakyReLU, regularizers, K
        if 'Sequential' not in globals():
            print('importing Keras from tensorflow...')
            #change all tensorflow to tf
            from tensorflow.compat.v1.keras.models import Sequential
            from tensorflow.compat.v1.keras.layers import Dense, Dropout, Activation, Flatten, Lambda
            from tensorflow.compat.v1.keras.layers import Conv2D, MaxPooling2D
            from tensorflow.compat.v1.keras.layers import LeakyReLU
            from tensorflow.compat.v1.keras.models import load_model
            from tensorflow.compat.v1.keras import regularizers
            from tensorflow.compat.v1.keras import backend as K

        self.image_size = image_size
        self.num_channels = image_channel
        self.num_labels = 10

        model = Sequential()
        tm = Flatten(input_shape=(image_size, image_size, image_channel))
        model.add(tm)
        # list of all hidden units weights
        self.U = []
        n = 0
        for param in params:
            n += 1
            # add each dense layer, and save a reference to list U
            self.U.append(Dense(param, kernel_initializer='he_uniform',
                          kernel_regularizer=regularizers.l2(l2_reg)))
            model.add(self.U[-1])
            # ReLU activation
            # model.add(Activation(activation))
            if activation == "arctan":
                model.add(Lambda(lambda x: tf.atan(x),
                          name=activation+"_"+str(n)))
            elif activation == "leaky":
                print("Leaky ReLU slope: {:.3f}".format(activation_param))
                model.add(LeakyReLU(alpha=activation_param,
                          name=activation+"_"+str(n)))
            else:
                model.add(Activation(activation, name=activation+"_"+str(n)))
            if dropout_rate > 0.0:
                model.add(Dropout(dropout_rate))
        self.W = Dense(10, kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(l2_reg))
        model.add(self.W)
        # output log probability, used for black-box attack
        if use_softmax:
            model.add(Activation('softmax'))
        if restore:
            model.load_weights(restore)

        layer_outputs = []
        # save the output of intermediate layers
        for layer in model.layers:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                layer_outputs.append(K.function(
                    [model.layers[0].input], [layer.output]))

        # a tensor to get gradients
        #tf.disable_eager_execution()
        import tensorflow
        self.gradients = []
        for i in range(model.output.shape[1]):
            output_tensor = model.output[:, i]
            self.gradients.append(tensorflow.keras.backend.gradients(output_tensor, model.input)[0])

        self.layer_outputs = layer_outputs
        self.model = model
        model.summary()

    def predict(self, data):
        return self.model(data)

    def get_gradient(self, data, sess= None):
        if sess is None:
            sess = K.get_session()
        # initialize all un initialized variables
        # sess.run(tf.variables_initializer([v for v in tf.global_variables() if v.name.split(':')[0] in set(sess.run(tf.report_uninitialized_variables()))]))
        evaluated_gradients = []
        for g in self.gradients:
            evaluated_gradients.append(sess.run(g, feed_dict={self.model.input: data}))
        return evaluated_gradients
