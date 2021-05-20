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
