from HS_common import *
from Args.argument import get_args

from keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D, Reshape, Dense, multiply, Permute, Concatenate, Conv3D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid
from keras.models import Model

from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.convolutional import MaxPooling3D, AveragePooling3D
from keras.layers import Dropout, Input
from keras.layers import Flatten, add
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization  # batch Normalization for managing internal covariant shift.
from keras.layers import Activation
from keras.utils import plot_model


def makeHS_3DCNN_5124(foldNum,modeltype,fc1,fc2):

    # Number of Convolutional Filters to use
    NumFilter = [5,10,20,40,60]

    # Convolution Kernel Size
    kernel_size = [3,3,3]
    stride_size = (1,1,1)
        
    if((SETT == 'SIG')or(SETT=='DUO')or(SETT=='BAL')or(SETT=='FIV')):
        input_shape = (imgRow, imgCol, imgDepth,1)

    if(modeltype >= 1):

        model = Sequential()
        ## convolutional layers
        model.add(Conv3D(filters=NumFilter[0], kernel_size = kernel_size, strides=stride_size, padding='same', input_shape=input_shape,
                        activation='relu',kernel_regularizer = regularizers.l2(Param_regularizers),kernel_initializer = initializers.he_uniform(seed=KERNEL_SEED)))
        ## add max pooling to obtain the most imformatic features
        model.add(MaxPool3D(pool_size=(2, 2, 2)))

        
        model.add(Conv3D(filters=NumFilter[1], kernel_size = kernel_size, strides=stride_size, padding='same',
                        activation='relu',kernel_regularizer = regularizers.l2(Param_regularizers),kernel_initializer = initializers.he_uniform(seed=KERNEL_SEED)))
        model.add(MaxPool3D(pool_size=(2, 2, 2)))

        
        model.add(Conv3D(filters=NumFilter[2], kernel_size = kernel_size, strides=stride_size, padding='same',
                        activation='relu',kernel_regularizer = regularizers.l2(Param_regularizers),kernel_initializer = initializers.he_uniform(seed=KERNEL_SEED)))                
        model.add(MaxPool3D(pool_size=(2, 2, 2)))

        
        model.add(Conv3D(filters=NumFilter[3], kernel_size = kernel_size, strides=stride_size, padding='same',
                        activation='relu',kernel_regularizer = regularizers.l2(Param_regularizers),kernel_initializer = initializers.he_uniform(seed=KERNEL_SEED)))                
        model.add(MaxPool3D(pool_size=(2, 2, 2)))

        
        # perform batch normalization on the convolution outputs before feeding it to MLP architecture
        model.add(Flatten())

        ## add BatchNorm and Dropout to avoid overfitting / perform regularization
        model.add(Dense(units = fc1, kernel_initializer = initializers.he_uniform(seed=KERNEL_SEED), kernel_regularizer = regularizers.l2(Param_regularizersDense)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(droprate))

        model.add(Dense(units = fc2, kernel_initializer = initializers.he_uniform(seed=KERNEL_SEED), kernel_regularizer = regularizers.l2(Param_regularizersDense)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(droprate))

        model.add(Dense(units = nb_classes,kernel_regularizer = regularizers.l2(Param_regularizersDense)))
        model.add(Activation('softmax'))

    # Compiling the CNN
    if(MULTI_CHECK):
        model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    else :
        model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
