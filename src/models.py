'''
This file contains defintions for the used models.
The models proposed by Zaid et al. all start with 'zaid_' and were taken from:
https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA

The simplified models in which the first convolutional layer is removed start with 'noConv1_'.
'''

from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
from keras import backend as K
from keras.optimizers import Adam


def zaid_ascad_desync_0(input_size=700,learning_rate=0.00001,classes=256):
	# Designing input layer
    input_shape = (input_size,1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_norm1')(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
        
    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
        
    # Logits layer              
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='ascad')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def zaid_ascad_desync_50(input_size=700,learning_rate=0.00001,classes=256):
    # Personal design
    input_shape = (input_size,1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(32, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_norm1')(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    # 2nd convolutional block
    x = Conv1D(64, 25, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(25, strides=25, name='block2_pool')(x)

    # 3rd convolutional block
    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv1')(x)      
    x = BatchNormalization()(x)
    x = AveragePooling1D(4, strides=4, name='block3_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification part
    x = Dense(15, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)

    # Logits layer
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='cnn_best')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def zaid_ascad_desync_100(input_size=700,learning_rate=0.00001,classes=256):
    # Personal design
    input_shape = (input_size,1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(32, 1, kernel_initializer='he_uniform', activation='linear', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_norm1')(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    # 2nd convolutional block
    x = Conv1D(64, 50, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(50, strides=50, name='block2_pool')(x)

    # 3rd convolutional block
    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv1')(x)      
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification part
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)

    # Logits layer
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='cnn_best')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def zaid_dpav4(input_size=4000,learning_rate=0.00001,classes=256):
    # Designing input layer
    input_shape = (input_size,1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_norm1')(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)

    # Logits layer
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='dpacontest_v4')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def zaid_aes_rd(input_size=700,learning_rate=0.00001,classes=256):
    # Designing input layer
    input_shape = (input_size,1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(8, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_norm1')(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    # 2nd convolutional block
    x = Conv1D(16, 50, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(50, strides=50, name='block2_pool')(x)

    # 3rd convolutional block
    x = Conv1D(32, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(7, strides=7, name='block3_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)

    # Logits layer      
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='aes_rd')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def zaid_aes_hd(input_size=1250,learning_rate=0.00001,classes=256):
    # Designing input layer
    input_shape = (input_size,1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_norm1')(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)

    # Logits layer
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='aes_hd_model')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def noConv1_ascad_desync_0(input_size=700,learning_rate=0.00001,classes=256):
    trace_input = Input(shape=(input_size,1))
    x = AveragePooling1D(2, strides=2, name='initial_pool')(trace_input)
    x = Flatten(name='flatten')(x)

    x = Dense(10, activation='selu', name='fc1')(x)
    x = Dense(10, activation='selu', name='fc2')(x)          
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(trace_input, x, name='noConv1_ascad_desync_0')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def noConv1_ascad_desync_50(input_size=700,learning_rate=0.00001,classes=256):
    trace_input = Input(shape=(input_size,1))
    x = AveragePooling1D(2, strides=2, name='initial_pool')(trace_input)

    x = Conv1D(64, 25, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(25, strides=25, name='block1_pool')(x)

    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv')(x)      
    x = BatchNormalization()(x)
    x = AveragePooling1D(4, strides=4, name='block2_pool')(x)

    x = Flatten(name='flatten')(x)

    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(trace_input, x, name='noConv1_ascad_desync_50')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def noConv1_ascad_desync_100(input_size=700,learning_rate=0.00001,classes=256):
    trace_input = Input(shape=(input_size,1))
    x = AveragePooling1D(2, strides=2, name='initial_pool')(trace_input)

    x = Conv1D(64, 50, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(50, strides=50, name='block2_pool')(x)

    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv')(x)      
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)

    x = Flatten(name='flatten')(x)

    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(trace_input, x, name='noConv1_ascad_desync_100')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def noConv1_dpav4(input_size=4000,learning_rate=0.00001,classes=256):
    trace_input = Input(shape=(input_size,1))
    x = AveragePooling1D(2, strides=2, name='initial_pool')(trace_input)

    x = Flatten(name='flatten')(x)

    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(trace_input, x, name='noConv1_dpav4')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def noConv1_aes_rd(input_size=700,learning_rate=0.00001,classes=256):
    trace_input = Input(shape=(input_size,1))
    x = AveragePooling1D(2, strides=2, name='initial_pool')(trace_input)

    x = Conv1D(16, 50, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(50, strides=50, name='block1_pool')(x)

    x = Conv1D(32, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(7, strides=7, name='block2_pool')(x)

    x = Flatten(name='flatten')(x)

    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(trace_input, x, name='noConv1_aes_rd')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def noConv1_aes_hd(input_size=1250,learning_rate=0.00001,classes=256):
    trace_input = Input(shape=(input_size,1))
    x = AveragePooling1D(2, strides=2, name='initial_pool')(trace_input)

    x = Flatten(name='flatten')(x)

    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(trace_input, x, name='noConv1_aes_hd')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model    