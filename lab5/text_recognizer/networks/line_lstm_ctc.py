from boltons.cacheutils import cachedproperty
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, RepeatVector, Reshape, TimeDistributed, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.lenet import lenet
from text_recognizer.networks.misc import slide_window
from text_recognizer.networks.ctc import ctc_decode


def line_lstm_ctc(input_shape, output_shape, window_width=28, window_stride=14):
    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    num_windows = int((image_width - window_width) / window_stride) + 1
    if num_windows < output_length:
        raise ValueError(f'Window width/stride need to generate at least {output_length} windows (currently {num_windows})')

    image_input = Input(shape=input_shape, name='image')
    y_true = Input(shape=(output_length,), name='y_true')
    input_length = Input(shape=(1,), name='input_length')
    label_length = Input(shape=(1,), name='label_length')

    gpu_present = len(device_lib.list_local_devices()) > 1
    lstm_fn = CuDNNLSTM if gpu_present else LSTM

    # Your code should use slide_window and extract image patches from image_input.
    # Pass a convolutional model over each image patch to generate a feature vector per window.
    # Pass these features through one or more LSTM layers.
    # Convert the lstm outputs to softmax outputs.
    # Note that lstms expect a input of shape (num_batch_size, num_timesteps, feature_length).

    ##### Your code below (Lab 3)
    #print (input_data.shape)
    #input_data = Input(name='the_input', shape=(image_height, image_width, 1), dtype='float32')
    
    
    #image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    
    #conv1 = Conv2D(32, (1, window_stride), activation='relu')(image_reshaped)
    
    
    #conv2 = Conv2D(64, (1, window_stride), activation='relu')(conv1)
    #mp2d = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    ##image_reshaped = Reshape((image_height, image_width, 1))(mp2d)#(image_input)
    # (image_height, image_width, 1)

    image_patches = Lambda(
        slide_window,
        arguments={'window_width': window_width, 'window_stride': window_stride}
    )(image_reshaped)
    # (num_windows, image_height, window_width, 1)
    print('image_patches shape done')
    print(image_patches.shape)

    # Make a LeNet and get rid of the last two layers (softmax and dropout)
    #convnet = lenet((image_height, window_width, 1), (num_classes,))
    #convnet = KerasModel(inputs=convnet.inputs, outputs=convnet.layers[-2].output)
    #convnet_outputs = TimeDistributed(image_patches)(image_patches)
    # (num_windows, 128)
    
    conv1 = Conv2D(32, (1, window_stride), activation='relu')(image_patches)
    print('con1 shape: ', conv1)
    
    #lstm_output = Bidirectional(lstm_fn(lstm_dim, return_sequences=True))(conv_squeezed)
    # (num_windows, lstm_dim * 2)
    
    #lstm_output2 = Bidirectional(lstm_fn(lstm_dim, return_sequences=True))(lstm_output)
    # (num_windows, lstm_dim * 2)

    #softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(lstm_output2)
    
    
    lstm_output = Bidirectional(lstm_fn(128, return_sequences=True))(conv1)
    lstm_output2 = Bidirectional(lstm_fn(128, return_sequences=True))(lstm_output)
    # (num_windows, 128)

    softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(lstm_output2)
    # (num_windows, num_classes)
    ##### Your code above (Lab 3)

    input_length_processed = Lambda(
        lambda x, num_windows=None: x * num_windows,
        arguments={'num_windows': num_windows}
    )(input_length)

    ctc_loss_output = Lambda(
        lambda x: K.ctc_batch_cost(x[0], x[1], x[2], x[3]),
        name='ctc_loss'
    )([y_true, softmax_output, input_length_processed, label_length])

    ctc_decoded_output = Lambda(
        lambda x: ctc_decode(x[0], x[1], output_length),
        name='ctc_decoded'
    )([softmax_output, input_length_processed])

    model = KerasModel(
        inputs=[image_input, y_true, input_length, label_length],
        outputs=[ctc_loss_output, ctc_decoded_output]
    )
    return model

