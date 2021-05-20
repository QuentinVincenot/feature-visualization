import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class FeatureVisualizer:
  #--------------------------------------------------------------------------------
  #----- Constructor & initialization
  #--------------------------------------------------------------------------------
  
  def __init__(self, model):
    self.model = model
    self.layers = []
    # Register the layers of the model
    self.__register_layers()

  #--------------------------------------------------------------------------------
  #----- Model & layers accessors
  #--------------------------------------------------------------------------------

  def __register_layers(self):
    # Iterate over every layer of the model
    for idx, _ in enumerate(self.model.layers):
      # Register only convolutional and fully connected layers
      if 'Conv2D' in str(type(self.model.layers[idx])):
        self.layers.append({'layer_type': 'Conv2D', 'name': self.model.layers[idx].name, 'neurons': self.model.layers[idx].filters})
      elif 'Dense' in str(type(self.model.layers[idx])):
        self.layers.append({'layer_type': 'Dense', 'name': self.model.layers[idx].name, 'neurons': self.model.layers[idx].units})
  
  def get_layers(self):
    # Return the list of registered layers
    return self.layers
  
  def list_layers(self):
    # Display information about the registered layers of the model
    for layer in self.layers:
      print(layer)

  #--------------------------------------------------------------------------------
  #----- Convolution filters visualization
  #--------------------------------------------------------------------------------
  
  def visualize_filter(self, layer_name, filter_id):
    # Retrieve the corresponding weights of the convolutional layer
    weights, _ = self.model.get_layer(layer_name).get_weights()
    # Normalize the weights of the filters to display them in grayscale
    min_weight, max_weight = weights.min(), weights.max()
    weights = (weights - min_weight) / (max_weight - min_weight)
    # Compute the number of rows/columns depending on the number of channels in filters
    MAX_COLUMNS = 8
    if weights.shape[2] < MAX_COLUMNS:
      nb_rows, nb_cols = 1, weights.shape[2]
    else:
      nb_rows, nb_cols = weights.shape[2] // MAX_COLUMNS, MAX_COLUMNS
    # Display the different channels of the selected filter in the layer
    _, axs = plt.subplots(nb_rows, nb_cols, figsize=(nb_cols*2, nb_rows*2))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    print('Filter {}'.format(filter_id))
    for row in range(nb_rows):
      for col in range(nb_cols):
        channel_id = row*MAX_COLUMNS + col
        if nb_rows == 1:
          axs[col].imshow(weights[:, :, channel_id, filter_id], cmap='gray')
          axs[col].set_xticks([]) ; axs[col].set_yticks([])
        else:
          axs[row, col].imshow(weights[:, :, channel_id, filter_id], cmap='gray')
          axs[row, col].set_xticks([]) ; axs[row, col].set_yticks([])
    plt.show()

  #--------------------------------------------------------------------------------
  #----- Classes maximization/visualization
  #--------------------------------------------------------------------------------
  
  def maximize_class(self, class_id, iterations=100, learning_rate=250.0):
    # Initialize a random image as a starting point for class maximization
    img = (tf.random.uniform((1, 224, 224, 3)) - 0.5) * 0.25
    # Optimize the starting random image a specific number of iterations
    for iteration in range(iterations):
      with tf.GradientTape() as tape:
        # Watch the input image to later get gradients of the loss
        tape.watch(img)
        # Make a forward pass through the model to get the class prediction
        class_prediction = self.model(img)[:, class_id]
        # Loss function is the mean prediction for the specified class
        loss = tf.reduce_mean(class_prediction)
      # Compute gradients of the loss with respect to the input random image to optimize
      grads = tape.gradient(loss, img)
      # Normalize the gradients
      grads = tf.math.l2_normalize(grads)
      # Optimize the image by adding the computed gradients
      img += learning_rate * grads
    # Once the input image has been optimized to maximize the designated layer, we normalize it
    img = img.numpy()
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15
    # We center back the optimized image crop
    img = img[0, 5:-5, 5:-5, :]
    # We clip the optimized image values in [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)
    # And we finally convert back the optimized image values to RGB
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img
    
  def visualize_all_classes(self, classes_ids, iterations=100, learning_rate=250.0):
    # Retrieve the number of classes to generate an optimized image for
    number_of_classes = len(classes_ids)
    # Compute the number of rows/columns depending on the number of classes
    MAX_COLUMNS = 2
    if number_of_classes < MAX_COLUMNS:
      nb_rows, nb_cols = 1, number_of_classes
    else:
      nb_rows, nb_cols = number_of_classes // MAX_COLUMNS, MAX_COLUMNS
    # Display the images maximizing all the different classes
    _, axs = plt.subplots(nb_rows, nb_cols, figsize=(nb_cols*8, nb_rows*8))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for row in range(nb_rows):
      for col in range(nb_cols):
        class_id = classes_ids[row*MAX_COLUMNS + col]
        image_maximizing_class = self.maximize_class(class_id, iterations, learning_rate)
        print(class_id, end=' ')
        if nb_rows == 1:
          axs[col].imshow(image_maximizing_class)
          axs[col].set_xticks([]) ; axs[col].set_yticks([])
        else:
          axs[row, col].imshow(image_maximizing_class)
          axs[row, col].set_xticks([]) ; axs[row, col].set_yticks([])
    plt.show()
