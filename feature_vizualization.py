import tensorflow as tf
import numpy as np


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
