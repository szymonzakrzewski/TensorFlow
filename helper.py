import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
import pathlib

import matplotlib.image as mpimg
import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Make the createing of our model a little easier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy



def view_random_image(target_dir, target_class):
  target_folder = target_dir+target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder),1)
  print(random_image)

  # Read in the image and plot it
  img = mpimg.imread(target_folder + "/" + random_image[0]) # IMPORTANT!!!!
  plt.imshow(img)
  plt.title(target_class)
  plt.axis(False) # Hides axies
  print(f"Image shape: {img.shape}")

  return img

  # Plot the validation and trainig curves separately

def plot_loss(history):
  """
  Retrun separate loss curves
  """
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.figure(figsize=(10,7))
  plt.plot(epochs, loss, label='training loss')
  plt.plot(epochs, val_loss, label='validation loss')
  plt.title('loss')
  plt.xlabel("epochs")
  plt.legend()

  # Plot accuracy 
  plt.figure(figsize=(10,7))
  plt.plot(epochs, accuracy, label='training accuracy')
  plt.plot(epochs, val_accuracy, label='validation accuracy')
  plt.title('accuracy')
  plt.xlabel("epochs")
  plt.legend()

  # Create a function to import a image and resize it to be able tobe used with our model
def load_and_prep_image(filename, img_shape=224):
  """
  Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape, colour_channels)
  """
  import tensorflow as tf
  # Read in the image
  img = tf.io.read_file(filename=filename)
  # Decode the read file into a tensor
  img = tf.image.decode_image(img)
  # Resize the image
  img = tf.image.resize(img,[img_shape, img_shape])
  # rescale the image and get all values betweet 0 & 1
  img = img / 255.
  return img