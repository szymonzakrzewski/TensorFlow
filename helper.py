# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import zipfile
# import os
# import pathlib

# import matplotlib.image as mpimg


# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Make the createing of our model a little easier
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
# from tensorflow.keras import Sequential
# from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy

IMAGE_SHAPE = (224, 224)

def unzip_data(file_url):
  """
  Downloads data and unzips it
  
  Args:
    * file_url (str): url link to file
  """
  !wget file_url
  # Unzip tour data
  zip_ref = zipfile.ZipFile('10_food_classes_10_percent.zip', 'r')
  zip_ref.extractall()
  zip_ref.close()

def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates TensorBoard callback

  Args:
    * dir_name (str): directory name of an experiment
    * experiment_name (str): name of an experiment

  Returns:
    * callback (tf.keras.callbacks.TensorBoard)
  """
  import datetime
  import tensorflow as tf
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime('%Y%m%d-%H%M%e')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f"Saving logs to {log_dir}")
  return tensorboard_callback

def view_random_image(target_dir, target_class):
  """
  Plots random image from directory and returns it as numpy array

  Args:
    * target_dir (str): directory with images
    * target_class (str): folder name of the class

  Returns:
    * The image data (numpy.array)     
  """
  import random
  import matplotlib.pyplot as plt
  import os
  import matplotlib.image as mpimg

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

def plot_loss(history):
  """
  Plots separate loss curves

  Args:
    * history (tf.keras.callbacks.History): A history object

  Returns:
    * None
  """  
  import matplotlib.pyplot as plt  
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

def load_and_prep_image(filename, img_shape=224):
  """
  Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape, colour_channels)

  Args:
    * filename (str): image filename
    * img_shape (int): size of an image

  Returns:
    * image (Tensor): scaled (1/255.) tensor
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

def walk_trought_directory(dir_path):  
  """
  Walk trought directory and prints its content

  Args:
    * dir_path (str): directory path
  """
  import os
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")

def create_model(model_url, num_classes=10):
  """
  Takes a TF Hub Url and craetes a Keras Sequential model with it.

  Args:
    model_url (str): A TensorFlow Hub feature extraction URL.
    num_classes (int): Number of putput neurons in the output layer,
    should be equal to number of target classes, default 10.

  Returns:
    An uncompiled Keras Sequantial model with model_url as feature extractor 
    layer and Dense output layer with num_classes output neurons.
  """
  import tensorflow_hub as hub
  from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
  from tensorflow.keras import Sequential
  feature_extractor_layer = hub.KerasLayer(handle=model_url, 
                                           trainable=False, # freeze the already learned patterns
                                           name='feature_extractor_layer',
                                           input_shape=IMAGE_SHAPE+(3,))
  model = Sequential([
      feature_extractor_layer,
      Dense(num_classes, activation='softmax', name='output_layer')
  ])
  return model





