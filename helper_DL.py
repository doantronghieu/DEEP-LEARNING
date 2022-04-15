"""
!wget https://raw.githubusercontent.com/doantronghieu/DEEP-LEARNING/main/helper_DL.py
!pip install colorama
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':15})
import seaborn           as sns
sns.set()
import helper_DL as helper
"""

### We create a bunch of helpful functions throughout the course.
### Storing them here so they're easily accessible.

import imp
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,\
                            recall_score, f1_score, precision_recall_fscore_support, \
                            classification_report    
                            
import tensorflow        as tf
import tensorflow.keras  as tfk
import tensorflow.keras.layers as tfkl
from tensorflow.keras import layers, optimizers, losses, models
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental import  preprocessing as PP
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import itertools, datetime, zipfile, os, random

from colorama import Fore
from google.colab import files
from keras.preprocessing import image

#=============================================================================#
FIG_SIZE = (22, 8)
IMG_SIZE = 224
BATCH_SIZE = 32
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] # Batch, Height, Width, Channels
OUTPUT_SHAPE = 1 # Not true, just for testing
MODEL_URL = 'https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5'
#=============================================================================#
def unzip_data(filename):
  """
  Unzips filename into the current working directory.
  Function to unzip a zipfile into current working directory 
  (since we're going to be downloading and unzipping a few files)

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()
#=============================================================================#
def walk_through_dir(dir_path):
  """
  Walk through an image classification directory and find out how many files (images)
    are in each subdirectory.
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk('dir_path'):
    print(f"{Fore.RED}{len(dirnames):>4} directories {Fore.GREEN}- {len(filenames):>5} files {Fore.BLUE}<== '...{dirpath[len('dir_path'):]}'.")
#=============================================================================#
def plot_correlation_matrix(data_frame, figsize=FIG_SIZE):
    corr_matrix = data_frame.corr()
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(data=corr_matrix, annot=True, linewidths=0.5, fmt='.2f', cmap='YlGnBu')
    plt.xticks(rotation=45)
    plt.show()
#=============================================================================#
def load_and_prep_image(filename, img_shape=IMG_SIZE, scale=True):
    """
    - Function for preprocessing images
    - Reads in an image from `filename`, turns it into a `Tensor` and reshapes it into (img_shape, img_shape, 3)
    to be able to used with the model

    Parameters 
    ----------
        filename  (str):  string filename of target image (filepath)
        img_shape (int):  size to resize the target image to
        scale     (bool): whether to scale pixel values to range(0, 1)
    """

    # Reads in the image file
    img = tf.io.read_file(filename)      
    # Decode the image into numerical Tensor with 3 colour channels (RGB)                 
    img = tf.image.decode_image(img, channels=3) 
    # Resize the image to the desired value
    img = tf.image.resize(img, [img_shape, img_shape])

    # Convert the colour channel values from 0-255 to 0-1
    if scale:
        return img / 255.
        # img = tf.image.convert_image_dtype(img, tf.float32)
    else:
        return img
#=============================================================================#
def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    """
    Description:
    - Creates batches of data out of image (X) and label (y) pairs.
    - Shuffles the data if it's training data, doesn't shuffle if it's validation data.
    - Accepts test data as input (no labels).

    Params:
    - X: Image filepath
    - y: Labels
    """

    if test_data:
        print('Creating test data batches ...')
        data = tf.data.Dataset.from_tensor_slices(tensors=(tf.constant(X)))
        data_batch = data.map(load_and_prep_image).batch(BATCH_SIZE)
        return data_batch
    elif valid_data:
        print('Creating validation data batches ...')
        data = tf.data.Dataset.from_tensor_slices(tensors=(tf.constant(X), tf.constant(y)))    
        data_batch = data.map(get_image_label_tuple).batch(BATCH_SIZE)
        return data_batch
    else:
        print('Creating training data batches ...')
        # Turns filepaths and labels into Tensors
        data = tf.data.Dataset.from_tensor_slices(tensors=(tf.constant(X), tf.constant(y)))
        # Shuffles pathnames and labels. Shuffles all the data
        data = data.shuffle(buffer_size=len(X))
        # Create (image, label) tuples (Also turns the image path into a preprocessed image)
        data = data.map(get_image_label_tuple)
        # Turn the training data into batches
        data_batch = data.batch(BATCH_SIZE)
    
    print('Creating data batches compeleted!')
    return data_batch
#=============================================================================#
def get_image_label_tuple(image_path, label):
    """
    - Takes an image file path name and the associated label,
    processes the image and returns a tuple of (image, label)
    """

    image = load_and_prep_image(image_path)

    return (image, tf.constant(label))
#=============================================================================#
def show_images_batch(images, labels, num_imgs_square, figsize, class_names):
    """
    - Displays a plot of `num_imgs_square` images and their labels from a data batch.
    - Usage: 
        train_images, train_labels = next(train_data.as_numpy_iterator())
        show_images_batch(images=train_images, 
                  labels=train_labels, 
                  num_imgs_square=5, 
                  figsize=15, 
                  class_names=unique_breeds)
    """

    plt.figure(figsize=(figsize, figsize))

    for i in range(num_imgs_square ** 2):
        ax = plt.subplot(num_imgs_square, num_imgs_square, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i].argmax()], fontsize=figsize-2)
        plt.axis('off')
#=============================================================================#
def unbatchify(batch_dataset, class_names):
    """
    - Takes a batched dataset of (image, label) Tensors and returns separate
    arrays of image and label.
    """

    images, labels = [], []

    # Loop through unbatched data
    for (image, label) in batch_dataset.unbatch().as_numpy_iterator():
        images.append(image)
        labels.append(class_names[np.argmax(label)])
    
    return (images, labels)
#=============================================================================#
def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
    print(f'Building model with: {MODEL_URL}')

    # Set up the model layers
    model = tfk.Sequential([
        hub.KerasLayer(MODEL_URL),
        tfk.layers.Dense(units=OUTPUT_SHAPE, activation='softmax')                            
    ])

    # Compile the model
    model.compile(loss=tfk.losses.CategoricalCrossentropy(),
                  optimizer=tfk.optimizers.Adam(),
                  metrics=['accuracy'])
    
    # Build the model
    model.build(INPUT_SHAPE)
    model.summary()

    return model
#=============================================================================#
from colorama import Fore
def check_layers(model):
    layer_idx_col      = 'No.'
    layer_name         = 'Name'
    layer_trainable    = 'Trainable'
    layer_dtype        = 'dtype'
    layer_dtype_policy = 'dtype policy'
    layer_output_shape = 'Output Shape'

    horizontal_bar = '-' * 110

    print(f'{Fore.BLACK}{layer_idx_col:^3}|{Fore.GREEN}{layer_name:^30}|{Fore.RED}{layer_trainable:^10}|{Fore.BLUE}{layer_dtype:^10}|{Fore.YELLOW}{layer_dtype_policy:^25}|{Fore.MAGENTA}{layer_output_shape:^30}')
    print(Fore.BLACK, horizontal_bar)
    for (layer_idx, layer_i) in enumerate(model.layers):
        print(f'{Fore.BLACK}{layer_idx:^3}|{Fore.GREEN}{layer_i.name:^30}|{Fore.RED}{layer_i.trainable:^10}|{Fore.BLUE}{layer_i.dtype:^10}|{Fore.YELLOW}{str(layer_i.dtype_policy):^25}|{Fore.MAGENTA}{str(layer_i.output_shape):^30}')
#=============================================================================#
def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instance to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name:        target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback
#=============================================================================#
def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(   confusion_matrix(y_test, y_preds),
                        annot=True, # Annotate the boxes
                        cbar=True  )
    plt.xlabel("Predicted label") # Predictions go on the x-axis
    plt.ylabel("True label") # True labels go on the y-axis 
#=============================================================================#
def plot_time_series(timesteps, values, y_label, format='.',  
                      start=0, end=None, label=None):
    """
    Plots timesteps (a series of points in time) againts values (a series of
    values across timesteps)

    Parameters:
        timesteps: array of timestep values
        values: array of values across time
        format: style of plot, default '.'
        start: where to start the plot (setting the value will index from start
                of timestep)
        end: where to end the plot
        label: label to show on plot about values
    """
    
    # Plot the series
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel('Time')
    plt.ylabel(y_label)
    if label:
        plt.legend(fontsize=14)
#=============================================================================#        
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """ 
  The following confusion matrix code is a remix of Scikit-Learn's 
  Makes a labelled confusion matrix comparing predictions and ground truth labels.
  If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

  Args:
    y_true:    Array of truth labels (must be same shape as y_pred).
    y_pred:    Array of predicted labels (must be same shape as y_true).
    classes:   Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize:   Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm:      normalize values or not (default=False).
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
  cm        = confusion_matrix(y_true, y_pred)
  cm_norm   = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  # Colors will represent how 'correct' a class is, darker == better
  cax     = ax.matshow(cm, cmap=plt.cm.Blues) 
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set( title="Confusion Matrix",
          xlabel="Predicted label",
          ylabel="True label",
          xticks=np.arange(n_classes), # create enough axis slots for each class
          yticks=np.arange(n_classes), 
          xticklabels=labels, # axes will labeled with class names (if they exist) or ints
          yticklabels=labels  )
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # x-labels vertically
  plt.xticks(rotation=70, fontsize=text_size)
  plt.yticks(fontsize=text_size)

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
#=============================================================================#
def predict_binary(target_size, model, class_names):
  """
  Args: 
    target_size (list)  (int)
    class_names (tuple) (str)
  """
  from google.colab import files
  from keras.preprocessing import image
  
  uploaded = files.upload()

  for fn in uploaded.keys():

      # Predicting images
      path = '/content/' + fn

      img = image.load_img(path, target_size=(target_size[0], target_size[1]))
      img = image.img_to_array(img)
      img = img / 255
      img = np.expand_dims(img, axis=0)

      images = np.vstack([img])
      classes = model.predict(images, batch_size=10)
      
      print(classes[0])
      
      if (classes[0] > 0.5):
          print(f'There is a(n) {class_names[0]} in the image')
      else:
          print(f'There is a(n) {class_names[1]} the in image')
#=============================================================================#
def get_pred_label(prediction_probabilities, class_names):
    """
    - Turns an array of prediction probabilities into a label.
    """

    return class_names[np.argmax(prediction_probabilities)]
#=============================================================================#
def pred_and_plot(model, filename, class_names):
  """
  Function to predict on images and plot them (works with multi-class)
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
#=============================================================================#
def make_preds(figsize, num_imgs, test_dir, model, class_names):

    for i in range(num_imgs):
        plt.figure(figsize=(figsize[0], figsize[1]))

        # Choose a random image from a random class(es)
        class_name = random.choice(class_names)
        filename   = random.choice(os.listdir(test_dir + '/' + class_name))
        filepath   = test_dir + class_name + '/' + filename

        # Load the image and make predictions
        img          = load_and_prep_image(filename=filepath, img_shape=224, scale=False)
        img_expanded = tf.expand_dims(img, axis=0)
        pred_prob    = model.predict(img_expanded)     # Get prediction probabilities array
        pred_class   = class_names[pred_prob.argmax()] # Get highest prediction probability index

        # Plot the image(s)
        plt.subplot(num_imgs, 2, i+1)
        plt.imshow(img / 255.)
        if (class_name == pred_class):
            title_color = 'g'
        else:
            title_color = 'r'
        plt.title(f'Actual: {class_name}. Pred: {pred_class}. Prob: {pred_prob.max():.2f}', 
                c=title_color)
        plt.axis(False);
#=============================================================================#
def plot_pred(prediction_probabilities, labels, images, class_names, index=1):
    """
    - Views the prediction, ground truth and image for sample n
    """

    pred_prob  = prediction_probabilities[index]
    true_label = labels[index]
    image      = images[index]
    pred_label = get_pred_label(pred_prob, class_names)

    if (pred_label == true_label):
        color = 'green'
    else:
        color = 'red'

    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{pred_label} ({np.max(pred_prob) * 100:.0f})%. True: {true_label}', color=color)
#=============================================================================#
def evaluate_preds(y_true, y_preds):
    """
    - Performs evaluation comparison on y_true labels vs. y_preds labels on a
    classification.
    """
    
    accuracy    = accuracy_score(y_true, y_preds)
    precision   = precision_score(y_true, y_preds)
    recall      = recall_score(y_true, y_preds)
    f1          = f1_score(y_true, y_preds)
    metric_dict = { 'Accuracy':  round(accuracy, 2),
                    'Precision': round(precision, 2),
                    'Recall':    round(recall, 2),
                    'F1':        round(f1, 2)   }
    
    print(f' Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'   Recall: {recall:.2f}')
    print(f' F1 score: {f1:.2f}')
    
    return metric_dict
#=============================================================================#
def plot_most_wrong_predict(test_data, test_dir, y_true, y_pred, preds_probs, class_names, 
                            model, images_to_view=9, start_index=0):
  """
  y_labels = []
  for images, labels in test_data.unbatch():
      y_labels.append(labels.numpy().argmax()) # argmax <= One-hot encoded labels

  preds_probs  = model.predict(test_data, verbose=1)
  pred_classes = tf.argmax(preds_probs, axis=1)
  class_names  = test_data.class_names
  """
  
  # 1. Get all of the image file paths in the test dataset
  filepaths = []

  for filepath in test_data.list_files(f'{test_dir}*/*.jpg',
                                      shuffle=False):
      filepaths.append(filepath.numpy()) # Convert to string
  
  # 2. Create a DataFrame of different parameters for each of out test images
  # Unravel test_data BatchDataset to get the test labels
  
  pred_df = pd.DataFrame({'img_path': filepaths,
                          'y_true': y_true,
                          'y_pred': y_pred,
                          'pred_conf': preds_probs.max(axis=1),
                          'y_true_classname': [class_names[i] for i in y_true],
                          'y_pred_classname': [class_names[i] for i in y_pred]})
  
  # 3. Find out in the DataFrame which predictions are wrong
  pred_df['pred_correct'] = pred_df['y_true'] == pred_df['y_pred']
  
  # 4. Sort the DataFrame to have most wrong predictions at the top
  top_100_wrong = pred_df[pred_df['pred_correct'] == False].sort_values('pred_conf', ascending=False)
  
  # 5.Visualize the test data samples which have the wrong prediction but highest pred probability
  plt.figure(figsize=(15, 15))

  for (i, row) in enumerate(top_100_wrong[start_index:start_index+images_to_view].itertuples()):
      # print(row)
      plt.subplot(3, 3, i+1)
      img = load_and_prep_image(filename=row[1], scale=False)
      _, _, _, _, pred_prob, y_true_classname, y_pred_classname, _ = row
      plt.imshow(img/255.)
      plt.title(f'Actual: {y_true_classname}\nPred: {y_pred_classname}\nProb: {pred_prob:.2f}')
      plt.axis(False)
#=============================================================================#
def plot_history_curves(history):
  """
  - Retrieve a list of list results on training and test data
  sets for each training epoch
  - Plot the validation and training data separately
  - Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object
  """ 
  loss         = history.history['loss']
  val_loss     = history.history['val_loss']
  accuracy     = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']
  
  epochs       = range(len(history.history['loss'])) # Get number of epochs

  plt.figure(figsize=(22, 8))
  
  # Plot loss
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss,     label='Training Loss')
  plt.plot(epochs, val_loss, label='Validation Loss')
  plt.title('Training and validation Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.subplot(1, 2, 2)
  plt.plot(epochs, accuracy,     label='Training Accuracy')
  plt.plot(epochs, val_accuracy, label='Validation Accuracy')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.legend();
#=============================================================================#
def compare_histories(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow History objects.

    Args:
      original_history: History object from original model (before new_history)
      new_history:      History object from continued model training (after original_history)
      initial_epochs:   Number of epochs in original_history (new_history plot starts from here) 
    """

    # Get original history metrics
    org_acc      = original_history.history['accuracy']
    org_loss     = original_history.history['loss']
    org_val_acc  = original_history.history['val_accuracy']
    org_val_loss = original_history.history['val_loss']

    # Get new history metrics
    new_acc      = new_history.history['accuracy']
    new_loss     = new_history.history['loss']
    new_val_acc  = new_history.history['val_accuracy']
    new_val_loss = new_history.history['val_loss']

    # Combine original history metrics with new history metrics
    total_acc      = org_acc      + new_acc
    total_loss     = org_loss     + new_loss
    total_val_acc  = org_val_acc  + new_val_acc
    total_val_loss = org_val_loss + new_val_loss

    # Make plots
    plt.figure(figsize=(22, 8))

    plt.subplot(1, 2, 1)
    plt.plot(total_acc,     label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning') # Zero indexed
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(total_loss,     label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning') # Zero indexed
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')

    plt.show();
#=============================================================================#
def calculate_results(y_true, y_pred):
  """
  Function to evaluate: accuracy, precision, recall, f1-score
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
    y_true: true labels in the form of a 1D array
    y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred)
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": round(model_accuracy, 3),
                  "precision": round(model_precision, 3),
                  "recall":    round(model_recall, 3),
                  "f1":        round(model_f1, 3)}
  return model_results
#=============================================================================#
def plot_f1_score_per_class(y_true, y_pred, class_names):
    classification_report_dict = classification_report(y_true, y_pred, output_dict=True)
    class_f1_scores            = {} # Empty Dictionary

    # Loop through classification report dictionary items {class_number: {metrics}}
    for (k, v) in classification_report_dict.items():
        if (k == 'accuracy'): # Stop once we get to accuracy key
            break
        else:
            # Add class names and f1-scores to new Dict
            class_f1_scores[class_names[int(k)]] = v['f1-score']

    # Turn f1-scores into dataframe for visualization
    f1_scores = pd.DataFrame({'class_names': list(class_f1_scores.keys()),
                              'f1-score':    list(class_f1_scores.values())}).sort_values('f1-score', 
                                                                                          ascending=False)

    fig, ax = plt.subplots(figsize=(12, 24))
    scores  = ax.barh(range(len(f1_scores)), f1_scores['f1-score'].values)
    ax.set_yticks(range(len(f1_scores)))
    ax.set_yticklabels(f1_scores['class_names'])
    ax.set_xlabel('F1-score')
    ax.set_title(f'F1-scores for {f1_scores.shape[0]} different Classes')
    ax.invert_yaxis() # Revert the order
    plt.show();
#=============================================================================#
def plot_roc_curve(fpr, tpr):
    """
    - Plots a ROC curve given the False Positive rate (fpr) and True Positive rate (tpr)
    of a model
    """
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='orange', label='ROC')
    
    # Plot line with no predictive power (baseline)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='Guessing')
    
    # Plot perfect ROC curve
    plt.plot([0, 0], [0, 1], color='r', linestyle='--')
    plt.plot([0, 1], [1, 1], color='r', linestyle='--')
    
    # Customize the plot
    plt.xlabel('False Positive Rate (fpr)')
    plt.ylabel('True Postitive Rate (tpr)')
    plt.title('Receiver Operating Characteristic (ROC) curve')
    plt.legend()
    plt.show()
#=============================================================================#
def visualize_intermediate_layers_binary(train_class_dirs, train_class_names,
                                         target_size, model):
    """
    Args: 
        train_class_dirs  (list)
        train_class_names (list)
    """
    #   Define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model
    successive_outputs = [layer.output for layer in model.layers]
    visualization_model = models.Model(inputs=model.input,
                                       outputs=successive_outputs)

    # Prepare a random input image from the training set.
    class0_img_files = [os.path.join(train_class_dirs[0], fn) for fn in train_class_names[0]]
    class1_img_files = [os.path.join(train_class_dirs[1], fn) for fn in train_class_names[1]]

    img_path = random.choice(class0_img_files + class1_img_files)
    img = load_img(img_path, target_size=(target_size[0], target_size[1])) # This is a PIL image
    x = img_to_array(img)         # Numpy array with shape (target_size, 3)
    x = x.reshape((1,) + x.shape) # Numpy array with shape (1, target_size, 3)
    x = x / 255.0 # Scale by 1 / 255

    #   Run the image through the network, thus obtaining all intermediate
    # representations for this image
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so we can have them as part of the plot
    layer_names = [layer.name for layer in model.layers]

    # Display the representations
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        # Just do this for the conv/maxpool layers, not the fully-connected layers
        if (len(feature_map.shape) == 4):
            
            n_features = feature_map.shape[-1] # Number of features in feature map
            size       = feature_map.shape[1]  # The feature map has shape (1, size, size, n_features)

            # Tile the images in this matrix
            display_grid = np.zeros((size, size * n_features))

            # Postprocess the feature to be visually palatable
            for i in range(n_features):
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')

                # Tile each filter into this big horizontal grid
                display_grid[:, i * size : (i + 1) * size] = x

            # Display the grid
            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis');
#=============================================================================#

#=============================================================================#

#=============================================================================#

#=============================================================================#

#=============================================================================#

#=============================================================================#

#=============================================================================#

#=============================================================================#

#=============================================================================#

#=============================================================================#

#=============================================================================#

#=============================================================================#

#=============================================================================#

#=============================================================================#

#=============================================================================#