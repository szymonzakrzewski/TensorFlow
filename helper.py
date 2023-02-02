import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import zipfile
import os
import pathlib
import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Make the createing of our model a little easier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy

import tensorflow_hub as hub
import zipfile
import random

IMAGE_SHAPE = (224, 224)

def unzip_data(file_name):
  """
  Unzips file by its name
  
  Args:
    * file_name (str): name of the file to unzip
  """
  # Unzip tour data
  zip_ref = zipfile.ZipFile(file_name, 'r')
  zip_ref.extractall()
  zip_ref.close()

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.
  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.
  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.
  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  

  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()
  plt.xticks(rotation=70)

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")


def visualize_and_predict(images_directory, class_names, model, image_shape = (224,224)):
  """
  Method for visualizing random image and make prediction on it
  Args:
    * images_directory (str) : path to directory with images
    * class_names (list[str]) : list of image classes
    * model (tf.keras.Model) : model for predictions
    * image_shape (tuple) : shape of an image, default (224,224)
  """
  target_class = random.choice(class_names)
  target_dir = images_directory + target_class
  random_image = random.choice(os.listdir(target_dir))
  random_image_path = target_dir + "/" + random_image

  # Read the random image 
  img = mpimg.imread(random_image_path)

  # Predict the image
  prediction = model.predict(tf.expand_dims(tf.image.resize(img, size=image_shape), axis=0))
  pred_class = class_names[tf.argmax(prediction[0])]

  # Plot predicted class of and image
  plt.imshow(img)
  plt.title(f"Original random image from class: {target_class},\n Predicted class: {pred_class}")
  plt.axis(False)

def compare_histories(old_hist, new_hist, initial_epochs=5):
  """
  Compares two TensorFlow History objects by printing them on plot
  Args:
    * old_hist (tf.keras.callbacks.History) : old history callback
    * new_hist (tf.keras.callbacks.History) : new history callback
  """
  # Get original history measurments
  acc = old_hist.history['accuracy']
  loss = old_hist.history['loss']

  val_acc = old_hist.history['val_accuracy']
  val_loss = old_hist.history['val_loss']

  # Combine old hist with new hist metrics
  total_acc = acc + new_hist.history['accuracy']
  total_loss = loss + new_hist.history['loss']
  total_val_acc = val_acc + new_hist.history['val_accuracy']
  total_val_loss = val_loss + new_hist.history['val_loss']

  # Make plot accuracy
  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(total_acc, label='Training Accuracy')
  plt.plot(total_val_acc, label='Val Accuracy')
  plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start fine funing')
  plt.legend(loc="lower right")
  plt.title("Training and Validation Accuracy")

  # Make plot loss
  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 2)
  plt.plot(total_loss, label='Training Loss')
  plt.plot(total_val_loss, label='Val Loss')
  plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start fine funing')
  plt.legend(loc="upper right")
  plt.title("Training and Validation Loss")

def unzip_data(file_name):
  """
  Unzips file by its name
  
  Args:
    * file_name (str): name of the file to unzip
  """
  # Unzip tour data
  zip_ref = zipfile.ZipFile(file_name, 'r')
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

def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Read in an image from file, turn it into tensor, reshapes it

  Args:
    filename (str): path to target image
    img_shape (int): height/width dimension of target image size
    scale (bool): scale pixel values from 0-255 to 0-1 or not

  Returns:
    Image tensor of shape (img_size, img_size)
  """

  # Read in the image
  img = tf.io.read_file(filename)

  # Decode image into tensor
  img = tf.io.decode_image(img, channels=3)

  # Resize the image
  imt = tf.image.resize(img, size=[img_shape, img_shape])

  # Scale
  if scale:
    # rescale the image
    return img/255.
  else:
    return img

def walk_trought_directory(dir_path):  
  """
  Walk trought directory and prints its content

  Args:
    * dir_path (str): directory path
  """
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
  feature_extractor_layer = hub.KerasLayer(handle=model_url, 
                                           trainable=False, # freeze the already learned patterns
                                           name='feature_extractor_layer',
                                           input_shape=IMAGE_SHAPE+(3,))
  model = Sequential([
      feature_extractor_layer,
      Dense(num_classes, activation='softmax', name='output_layer')
  ])
  return model





