import tensorflow as tf
from tensorflow import keras
from keras import layers, activations
import os 
from dotenv import load_dotenv
load_dotenv()

WINDOW_SIZE=int(os.environ.get('WINDOW_SIZE'))
NUM_VARIABLES=int(os.environ.get('NUM_VARIABLES'))
NUM_CLASSES=int(os.environ.get('NUM_CLASSES'))

def AE_1():
    img_input1 = layers.Input(shape=(WINDOW_SIZE, NUM_VARIABLES))
    label = layers.Input(shape=(1,))

    x = layers.Conv1DTranspose(3, 5, strides=(1), activation = "linear", padding="valid", use_bias=True, dtype='float32')(img_input1)
    x = layers.MaxPool1D((2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(64, 32, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, 16, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(256, 8, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.MaxPool1D((2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(512, 3, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.MaxPool1D((2))(x)
    x = layers.BatchNormalization()(x)    
    x = layers.Dropout(0.3)(x)

    x= layers.Conv1D(1024, 3, strides=(1), activation = "relu", padding="same", use_bias=True)(x)    
    
    x = layers.Flatten()(x)
    
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.Concatenate()([x, layers.Flatten()(tf.one_hot(   tf.cast(label, tf.int32) , NUM_CLASSES))])
    
    x = layers.Dense(NUM_VARIABLES * 26, use_bias=False)(x)
    x = layers.Reshape((26 , NUM_VARIABLES))(x)

    x= layers.Conv1D(1024, 3, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D((2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(512, 3, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D((2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(256, 8, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, 16, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(64, 32, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D((2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(3, 5, strides=(1), activation = "linear", padding="valid", use_bias=True, dtype='float32')(x)

    g_model = keras.models.Model(inputs = [img_input1, label], outputs = x, name="generator")
    return g_model

def AE_2():
    img_input1 = layers.Input(shape=(WINDOW_SIZE, NUM_VARIABLES))
    label = layers.Input(shape=(1,))

    x = layers.Conv1DTranspose(3, 5, strides=(1), activation = "linear", padding="valid", use_bias=True, dtype='float32')(img_input1)
    x = layers.MaxPool1D((2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(32, 64, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(64, 32, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, 16, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.MaxPool1D((2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(256, 8, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.MaxPool1D((2))(x)
    x = layers.BatchNormalization()(x)    
    x = layers.Dropout(0.3)(x)

    x= layers.Conv1D(512, 4, strides=(1), activation = "relu", padding="same", use_bias=True)(x)    
    
    x = layers.Flatten()(x)
    
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.Concatenate()([x, layers.Flatten()(tf.one_hot(   tf.cast(label, tf.int32) , NUM_CLASSES))])
    
    x = layers.Dense(NUM_VARIABLES * 26, use_bias=False)(x)
    x = layers.Reshape((26 , NUM_VARIABLES))(x)

    x= layers.Conv1D(512, 4, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D((2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(256, 8, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D((2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, 16, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(64, 32, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(32, 64, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D((2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(3, 5, strides=(1), activation = "linear", padding="valid", use_bias=True, dtype='float32')(x)

    g_model = keras.models.Model(inputs = [img_input1, label], outputs = x, name="generator")
    return g_model

def AE_3():
    img_input1 = layers.Input(shape=(WINDOW_SIZE, NUM_VARIABLES))
    label = layers.Input(shape=(1,))

    x = layers.Conv1DTranspose(3, 5, strides=(1), activation = "linear", padding="valid", use_bias=True, dtype='float32')(img_input1)
    x = layers.MaxPool1D((2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(32, 32, strides=(1), padding="same", use_bias=True)(x)

    skip1_s = x    
    x = layers.Conv1D(32, 32, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(32, 32, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)

    skip1_e = layers.Add()([skip1_s,x])
    x = layers.Activation(activations.relu)(skip1_e)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPool1D((2))(x)
    x = layers.Conv1D(128, 8, strides=(1), padding="same", use_bias=True)(x)

    skip2_s = x
    x = layers.Conv1D(128, 8, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, 8, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)

    skip2_e = layers.Add()([skip2_s,x])
    x = layers.Activation(activations.relu)(skip2_e)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPool1D((2))(x)
    x = layers.Conv1D(512, 3, strides=(1), padding="same", use_bias=True)(x)

    skip3_s = x
    x = layers.Conv1D(512, 3, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(512, 3, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)

    skip3_e = layers.Add()([skip3_s,x])
    x = layers.Activation(activations.relu)(skip3_e)   
    x = layers.Dropout(0.3)(x)    
    x = layers.MaxPool1D((2))(x)

    x = layers.Flatten()(x)    
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.Concatenate()([x, layers.Flatten()(tf.one_hot(   tf.cast(label, tf.int32) , NUM_CLASSES))])    
    x = layers.Dense(NUM_VARIABLES * 26, use_bias=False)(x)
    x = layers.Reshape((26 , NUM_VARIABLES))(x)
    
    x = layers.Conv1D(512, 3, strides=(1), padding="same", use_bias=True)(x)
    skip4_s = x
    x = layers.Conv1D(512, 3, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(512, 3, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)

    skip4_e = layers.Add()([skip4_s,x])
    x = layers.Activation(activations.relu)(skip4_e) 
    x = layers.Dropout(0.3)(x)
    x = layers.UpSampling1D((2))(x)
    x = layers.Conv1D(128, 8, strides=(1), padding="same", use_bias=True)(x)

    skip5_s = x
    x = layers.Conv1D(128, 8, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(128, 8, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)

    skip5_e = layers.Add()([skip5_s,x])
    x = layers.Activation(activations.relu)(skip5_e)
    x = layers.Dropout(0.3)(x)
    x = layers.UpSampling1D((2))(x)
    x = layers.Conv1D(32, 32, strides=(1), padding="same", use_bias=True)(x)

    skip6_s = x    
    x = layers.Conv1D(32, 32, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(32, 32, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)

    skip6_e = layers.Add()([skip6_s,x])
    x = layers.Activation(activations.relu)(skip6_e)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D((2))(x)
    x = layers.Conv1D(3, 5, strides=(1), activation = "linear", padding="valid", use_bias=True, dtype='float32')(x)

    g_model = keras.models.Model(inputs = [img_input1, label], outputs = x, name="generator")
    return g_model

def AE_4():
    img_input1 = layers.Input(shape=(WINDOW_SIZE, NUM_VARIABLES))
    label = layers.Input(shape=(1,))

    x = layers.Conv1DTranspose(3, 5, strides=(1), activation = "linear", padding="valid", use_bias=True, dtype='float32')(img_input1)
    x = layers.MaxPool1D((2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(64, 64, strides=(1), padding="same", use_bias=True)(x)
    
    skip1_s = x    
    x = layers.Conv1D(64, 64, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, 64, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    
    skip1_e = layers.Add()([skip1_s,x])
    x = layers.Activation(activations.relu)(skip1_e)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPool1D((2))(x)
    x = layers.Conv1D(256, 16, strides=(1), padding="same", use_bias=True)(x)
    
    skip2_s = x
    x = layers.Conv1D(256, 16, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, 16, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    
    skip2_e = layers.Add()([skip2_s,x])
    x = layers.Activation(activations.relu)(skip2_e)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPool1D((2))(x)
    x = layers.Conv1D(512, 3, strides=(1), padding="same", use_bias=True)(x)
    
    skip3_s = x
    x = layers.Conv1D(512, 3, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(512, 3, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    
    skip3_e = layers.Add()([skip3_s,x])
    x = layers.Activation(activations.relu)(skip3_e)   
    x = layers.Dropout(0.3)(x)    
    x = layers.MaxPool1D((2))(x)    
    
    x = layers.Flatten()(x)    
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.Concatenate()([x, layers.Flatten()(tf.one_hot(   tf.cast(label, tf.int32) , NUM_CLASSES))])    
    x = layers.Dense(NUM_VARIABLES * 26, use_bias=False)(x)
    x = layers.Reshape((26 , NUM_VARIABLES))(x)    
    
    x = layers.Conv1D(512, 3, strides=(1), padding="same", use_bias=True)(x)
    
    skip4_s = x
    x = layers.Conv1D(512, 3, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(512, 3, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    
    skip4_e = layers.Add()([skip4_s,x])
    x = layers.Activation(activations.relu)(skip4_e) 
    x = layers.Dropout(0.3)(x)
    x = layers.UpSampling1D((2))(x)
    x = layers.Conv1D(256, 16, strides=(1), padding="same", use_bias=True)(x)
    
    skip5_s = x
    x = layers.Conv1D(256, 16, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(256, 16, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    
    skip5_e = layers.Add()([skip5_s,x])
    x = layers.Activation(activations.relu)(skip5_e)
    x = layers.Dropout(0.3)(x)
    x = layers.UpSampling1D((2))(x)
    x = layers.Conv1D(64, 64, strides=(1), padding="same", use_bias=True)(x)
    
    skip6_s = x    
    x = layers.Conv1D(64, 64, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, 64, strides=(1), padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    
    skip6_e = layers.Add()([skip6_s,x])
    x = layers.Activation(activations.relu)(skip6_e)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D((2))(x)
    x = layers.Conv1D(3, 5, strides=(1), activation = "linear", padding="valid", use_bias=True, dtype='float32')(x)

    g_model = keras.models.Model(inputs = [img_input1, label], outputs = x, name="generator")
    return g_model

def AE_5():
    img_input1 = layers.Input(shape=(WINDOW_SIZE, NUM_VARIABLES))
    label = layers.Input(shape=(1,))

    x = layers.LSTM(100,return_sequences = True, stateful=False)(img_input1)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(20)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, layers.Flatten()(tf.one_hot(   tf.cast(label, tf.int32) , NUM_CLASSES))])
    x = layers.Reshape((20 , 2))(x)

    x = layers.LSTM(20, return_sequences = True)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(100, stateful=False)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(WINDOW_SIZE*NUM_VARIABLES, activation="linear")(x)
    x = layers.Reshape((WINDOW_SIZE , NUM_VARIABLES))(x)

    g_model = keras.models.Model(inputs = [img_input1, label], outputs = x, name="generator")
    return g_model

def AE_6():
    img_input1 = layers.Input(shape=(WINDOW_SIZE, NUM_VARIABLES))
    label = layers.Input(shape=(1,))
    x = layers.Flatten()(img_input1)

    x = layers.Dense(360)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(160)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, layers.Flatten()(tf.one_hot(   tf.cast(label, tf.int32) , NUM_CLASSES))])   
    
    x = layers.Dense(160)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(360)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(WINDOW_SIZE*NUM_VARIABLES, activation="linear")(x)
    x = layers.Reshape((WINDOW_SIZE , NUM_VARIABLES))(x)

    g_model = keras.models.Model(inputs = [img_input1, label], outputs = x, name="generator")
    return g_model