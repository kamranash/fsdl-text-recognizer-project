from typing import Tuple

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D
import numpy as np

def mlp(input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        layer_size: int=128,
        dropout_amount: float=0.2,
        num_layers: int=3) -> Model:
    """
    Simple multi-layer perceptron: just fully-connected layers with dropout between them, with softmax predictions.
    Creates num_layers layers.
    """
    num_classes = output_shape[0]

    model = Sequential()
    
    
    # Don't forget to pass input_shape to the first layer of the model
    #print(input_shape)
    ##### Your code below (Lab 1)
    model.add(Reshape( (input_shape[0], input_shape[1], 1), input_shape=input_shape))
 
    #input_shape = np.reshape(input_shape, (input_shape[0], input_shape[1], 1))
    #model.add(Dense((input_shape[0]*input_shape[1]), activation='relu'))
    model.add(Conv2D(16, (4, 4), strides = (1, 1), padding='same', activation='relu' ))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(8, (3, 3), strides = (1, 1), activation='relu' ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(4, (2, 2), strides = (1, 1), activation='relu' ))
    #model.add(Flatten(input_shape=input_shape))  
    #model.add(Dense(input_shape[0]*input_shape[0]/2, activation='relu'))
    #model.add(Dense(input_shape[0]*input_shape[0], activation='relu'))
    #model.add(Dropout(dropout_amount))
    #model.add(Dense((input_shape[0]*input_shape[0])/2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(layer_size, activation='relu'))
    Dropout(dropout_amount)
    #model.add(Dense(layer_size/2, activation='relu'))
    
#    for _ in range(num_layers):
#        model.add(Dense(layer_size, activation='relu'))
#        model.add(Dropout(dropout_amount))
    model.add(Dense(num_classes, activation='softmax'))
    ##### Your code above (Lab 1)

    return model

