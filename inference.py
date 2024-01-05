import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import pandas as pd
import os 
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

params = sys.argv
print(params)
try:
    NUMBER_SAMPLES = int(params[1])
except:
    print("Error, the user has to set the number of synthetic samples to generate as parameter. ")
    exit(0)

WINDOW_SIZE=int(os.environ.get('WINDOW_SIZE'))
NUM_VARIABLES=int(os.environ.get('NUM_VARIABLES'))
NOISE_DIM=int(os.environ.get('NOISE_DIM'))
NUM_CLASSES=int(os.environ.get('NUM_CLASSES'))

generator = keras.models.load_model("generator.h5")

random_latent_vectors = tf.random.normal(shape=(NUMBER_SAMPLES, NOISE_DIM))
uniform_space = tf.cast(tf.random.uniform(shape=(NUMBER_SAMPLES, 1), minval=0, maxval=NUM_CLASSES, dtype=tf.int32), tf.float32)
random_latent_vectors = [random_latent_vectors , uniform_space ]
generated_images = generator(random_latent_vectors)
for i in range(NUMBER_SAMPLES):
    df = pd.DataFrame(generated_images[i].numpy().reshape((NUM_VARIABLES,WINDOW_SIZE)) , index=["x","y","z"])
    input = generated_images[i].numpy().reshape((NUM_VARIABLES,WINDOW_SIZE))
    input = input.reshape((NUM_VARIABLES,WINDOW_SIZE,1))
    input = input.reshape((1,NUM_VARIABLES,WINDOW_SIZE,1))
    label = np.array(uniform_space[i])
    df.T.plot()
    plt.savefig(f'output/{i}_activity_{int(uniform_space[i][0])}.png')