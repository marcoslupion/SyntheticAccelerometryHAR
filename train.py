import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import pandas as pd
import os 
from models import *
from dotenv import load_dotenv
load_dotenv()
WINDOW_SIZE=int(os.environ.get('WINDOW_SIZE'))
NUM_VARIABLES=int(os.environ.get('NUM_VARIABLES'))
NOISE_DIM=int(os.environ.get('NOISE_DIM'))
NUM_CLASSES=int(os.environ.get('NUM_CLASSES'))
CGAN= True if os.environ.get('CGAN') == "True" else False

print(WINDOW_SIZE)
print(NUM_VARIABLES)
print(NOISE_DIM)
print(NUM_CLASSES)
print(CGAN)

# Training logs
checkpoint_dir = f'checkpoints/'
callback_csv_filename = "training_evolution_metrics.csv"

# Models
g_model = generator()
d_model = discriminator()

# Training parameters
BATCH_SIZE = 16
epochs = 2
lr = 0.00005

class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=5,
        gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images, real_labels):
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images

        interpolated = real_images + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator([interpolated,real_labels] , training=True)
        grads = gp_tape.gradient(pred, [interpolated])
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    @tf.function
    def train_step(self, data):

        x, y = data
        real_images = [x, y]
        batch_size = tf.shape(x)[0]

        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, NOISE_DIM))
            uniform_space = y
            random_latent_vectors = [random_latent_vectors , uniform_space ]

            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator([fake_images,uniform_space], training=True)

                real_logits = self.discriminator(real_images, training=True)

                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)

                gp = self.gradient_penalty(batch_size, x, fake_images, y)
                d_loss = d_cost + gp * self.gp_weight

            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        random_latent_vectors = tf.random.normal(shape=(batch_size, NOISE_DIM))
        uniform_space = y
        random_latent_vectors = [random_latent_vectors , uniform_space ]

        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            gen_img_logits = self.discriminator([generated_images, uniform_space], training=True)
            g_loss = self.g_loss_fn(gen_img_logits)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "d_cost" : d_cost, "d_gp" : gp, "g_loss": g_loss, "d_real_loss" : tf.reduce_mean(real_logits),"d_fake_loss" : tf.reduce_mean(fake_logits) }


class cGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=5,
    ):
        super(cGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(cGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
    
    @tf.function
    def train_step(self, data):

        x, y = data

        real_images = [x, y]

        batch_size = tf.shape(x)[0]

        random_latent_vectors = tf.random.normal(shape=(batch_size, NOISE_DIM))
        uniform_space = y
        random_latent_vectors = [random_latent_vectors , uniform_space ]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(random_latent_vectors, training=True)

            real_logits = self.discriminator(real_images, training=True)

            fake_logits = self.discriminator([fake_images,uniform_space], training=True)
            
            gen_loss = self.g_loss_fn(fake_logits)
            disc_loss = self.d_loss_fn(real_logits, fake_logits) 

        d_gradient = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_gradient, self.discriminator.trainable_variables)
        )

        gen_gradient = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": disc_loss, "g_loss": gen_loss}

class SaveCheckpointCallback(keras.callbacks.Callback):
    def __init__(self, manager):
      super(SaveCheckpointCallback, self).__init__()
      self.manager = manager
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 50 == 0:
            self.manager.save()



if CGAN:
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr , beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr , beta_1=0.5, beta_2=0.9)
else:
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr , beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return real_loss - fake_loss

def generator_loss(fake_img):
    return tf.reduce_mean(fake_img)

def discriminator_loss_cgan(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss_cgan(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,discriminator_optimizer=discriminator_optimizer,generator = g_model, discriminator=d_model)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)
CSV_clb = tf.keras.callbacks.CSVLogger(callback_csv_filename, separator=";", append=False)
callbacks = [SaveCheckpointCallback(manager),CSV_clb]

print(g_model.summary())
print(d_model.summary())

if CGAN:
    wgan = cGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=NOISE_DIM,
        discriminator_extra_steps=5,
    )
else:
    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=NOISE_DIM,
        discriminator_extra_steps=5,
    )

if CGAN: 
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss_cgan,
        d_loss_fn=discriminator_loss_cgan,
    )
else:
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )

dataset_name = "dataset"
CHARACTER_SEPARATOR = "\\"
dict_files = {}
end = False
for path, subdirs, files in os.walk(dataset_name):
    lower_names = [f.lower() for f in files]
    for name in [f for f in lower_names if ".csv" in f]:
        filename = os.path.join(path, name)
        splitted = f'{path}{CHARACTER_SEPARATOR}{filename}'.split(CHARACTER_SEPARATOR)
        type_dataset = splitted[1]
        if type_dataset not in dict_files:
            dict_files[type_dataset] = {
                "registers" : [],
                "classes" : []
            }
        class_data = splitted[2]
        df = pd.read_csv(filename , header=None).T
        dict_files[type_dataset]["registers"].append(df.to_numpy().reshape((204,3)).astype('float32'))
        dict_files[type_dataset]["classes"].append(float(class_data))

test_ds = tf.data.Dataset.from_tensor_slices((dict_files["train"]["registers"] , dict_files["train"]["classes"])).shuffle(len(dict_files["train"]["classes"])).batch(BATCH_SIZE)

wgan.fit(test_ds, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[callbacks])

wgan.generator.save(f'generator.h5')
wgan.discriminator.save(f'discriminator.h5')