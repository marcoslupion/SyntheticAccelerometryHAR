import tensorflow as tf
from tensorflow import keras
from keras import layers
import os 
from dotenv import load_dotenv
load_dotenv()

WINDOW_SIZE=int(os.environ.get('WINDOW_SIZE'))
NUM_VARIABLES=int(os.environ.get('NUM_VARIABLES'))
NOISE_DIM=int(os.environ.get('NOISE_DIM'))
NUM_CLASSES=int(os.environ.get('NUM_CLASSES'))
IMG_SHAPE = (NUM_VARIABLES, WINDOW_SIZE)
CGAN= True if os.environ.get('CGAN') == "True" else False

def generator():
    noise = layers.Input(shape=(NOISE_DIM,))
    label = layers.Input(shape=(1,))
    x = layers.Concatenate()([noise, layers.Flatten()(tf.one_hot(tf.cast(label, tf.int32) , NUM_CLASSES))])

    x = layers.Dense(NUM_VARIABLES * 26, use_bias=False)(x)
    x = layers.Reshape((26 , NUM_VARIABLES))(x)

    x= layers.Conv1D(1024, 3, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D((2))(x)

    x = layers.Conv1D(512, 3, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D((2))(x)

    x = layers.Conv1D(256, 8, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, 16, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(64, 32, strides=(1), activation = "relu", padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D((2))(x)

    x = layers.Conv1D(3, 5, strides=(1), activation = "linear", padding="valid", use_bias=True, dtype='float32')(x)

    g_model = keras.models.Model(inputs = [noise, label], outputs = x, name="generator")
    return g_model

def discriminator():
    img_input1 = layers.Input(shape=(WINDOW_SIZE, NUM_VARIABLES))
    img_input2 = layers.Input(shape=(1,))
    img_input2 = tf.reshape(img_input2 , [-1])

    x = layers.Conv1D(filters=96, kernel_size=48, activation='relu' , strides = (12))(img_input1)    
    x = layers.Conv1D(filters=256, kernel_size=(8), strides = (1), activation='relu', padding = "same")(x)
    x = layers.Conv1D(filters=512, kernel_size=(6), strides = (1), activation='relu', padding = "valid")(x)
    x = layers.Conv1D(filters=512, kernel_size=(4), strides = (1), activation='relu', padding = "same")(x)    
    x = layers.Conv1D(filters=1024, kernel_size=(3), strides = (1), activation='relu', padding = "valid")(x)

    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, tf.one_hot(tf.cast(img_input2, tf.int32) , NUM_CLASSES)])

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(8, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation='linear', dtype='float32')(x)

    d_model = keras.models.Model(inputs=[img_input1,img_input2] , outputs = x, name="discriminator")
    return d_model