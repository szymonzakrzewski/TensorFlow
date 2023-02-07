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
from tensorflow.keras.layers.experimental import preprocessing


import tensorflow_hub as hub
import zipfile
import random

IMAGE_SHAPE = (224, 224)

def evaluation(y_true, y_pred):
  """
  Calculates Accuracy, Precision, Recall and F1 score
  of a binary classification model.

  Args:
    y_true (list): list of targets
    y_pred (list): list of predicted values

  Returns:
    Dictionary with Accuracy, Precision, Recall and F1 score values
  """
  from sklearn.metrics import f1_score

  m = tf.keras.metrics.Accuracy()
  m.update_state(y_true, y_pred)
  acc_res = m.result().numpy() * 100

  m = tf.keras.metrics.Precision()
  m.update_state(y_true, y_pred)
  prec_res = m.result().numpy() * 100

  m = tf.keras.metrics.Recall()
  m.update_state(y_true, y_pred)
  rec_res = m.result().numpy() * 100

  f1_res = f1_score(y_true, y_pred) * 100

  return {
      "accuracy": acc_res,
      "precision": prec_res,
      "recall": rec_res,
      "f1_score": f1_res
      }

def pred_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class
  if len(pred[0]) > 1: # check for multi-class
    pred_class = class_names[pred.argmax()] # if more than one output, take the max
  else:
    pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

  # Plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False);

def create_model_checkpoint_callback(filepath, monitor="val_loss", verbose=1, 
                                   save_best_only=True, save_weights_only=True,
                                   save_freq='epoch'):
  """
  Creates ModelCheckpoint callback

  Args:
    filepath: string or PathLike, path to save the model file. e.g.
      filepath = os.path.join(working_dir, 'ckpt', file_name). filepath
      can contain named formatting options, which will be filled the value
      of epoch and keys in logs (passed in on_epoch_end). For example:
      if filepath is weights.{epoch:02d}-{val_loss:.2f}.hdf5, then the
      model checkpoints will be saved with the epoch number and the
      validation loss in the filename. The directory of the filepath should
      not be reused by any other callbacks to avoid conflicts.
    monitor: The metric name to monitor. Typically the metrics are set by
      the Model.compile method. Note:
      Prefix the name with "val_" to monitor validation metrics.
      Use "loss" or "val_loss" to monitor the model's total loss.
      If you specify metrics as strings, like "accuracy", pass the same
      string (with or without the "val_" prefix).
      If you pass metrics.Metric objects, monitor should be set to
      metric.name
      If you're not sure about the metric names you can check the contents
      of the history.history dictionary returned by
      history = model.fit()
      Multi-output models set additional prefixes on the metric names.

    verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
      displays messages when the callback takes an action.
    save_best_only: if save_best_only=True, it only saves when the model
      is considered the "best" and the latest best model according to the
      quantity monitored will not be overwritten. If filepath doesn't
      contain formatting options like {epoch} then filepath will be
      overwritten by each new better model.
    save_weights_only: if True, then only the model's weights will be saved
      (model.save_weights(filepath)), else the full model is saved
      (model.save(filepath)).
    save_freq: 'epoch' or integer. When using 'epoch', the callback
      saves the model after each epoch. When using integer, the callback
      saves the model at end of this many batches. 
  Returns:
    ModelCheckpointCallback
  """
  callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    monitor=monitor,
    verbose=verbose,
    save_best_only=save_best_only,
    save_weights_only=save_weights_only,
    save_freq=save_freq
  )
  return callback

def create_early_stopping_callback(monitor='val_loss', min_delta=0, mode='auto', 
                                   patience=0, start_from_epoch=0):
  """
  Creates EarlyStopping callback

  Args:
  monitor: Quantity to be monitored.
  min_delta: Minimum change in the monitored quantity
      to qualify as an improvement, i.e. an absolute
      change of less than min_delta, will count as no
      improvement.
  patience: Number of epochs with no improvement
      after which training will be stopped.
  mode: One of {"auto", "min", "max"}. In min mode,
      training will stop when the quantity
      monitored has stopped decreasing; in "max"
      mode it will stop when the quantity
      monitored has stopped increasing; in "auto"
      mode, the direction is automatically inferred
      from the name of the monitored quantity.
  start_from_epoch: Number of epochs to wait before starting
      to monitor improvement. This allows for a warm-up period in which
      no improvement is expected and thus training will not be stopped.
  Returns:
    EarlyStoppingCallback
  """
  callback = tf.keras.callbacks.EarlyStopping(
      monitor=monitor,
      min_delta=min_delta,
      mode=mode,
      patience=patience,
      start_from_epoch=start_from_epoch
  )
  return callback


def preprocess_image(image, label, img_shape=224, scale=False):
  """
  Converts img datatype from 'uint8' -> 'float32' and reshapes
  image to [img_shape, img_shape, colour_channels]
  
  Args:
    image (tensor): image tensor
    label (tensor): class number
    img_shape (int): target image size
    scale (bool): scale by 255.
  """
  image = tf.image.resize(img, size=[img_shape, img_shape])
  if scale:
    image = image/225.
  return tf.cast(image, dtype=tf.float32), label # return tuple (float32_image, label)

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
  img = tf.image.resize(img, size=[img_shape, img_shape])

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





