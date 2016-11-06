import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class gameNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=4, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    C, H, W = input_dim
    F = num_filters

    # convoltional layer
    # W1 size of (num_filters, channels, filter_size, filter_size)
    # b1 size of H, W, num_filters, assume we always use pading that can keep
    # same size as input after conv layer
    self.params['W1'] = np.random.normal(
      0, weight_scale, (num_filters, input_dim[0], filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)

    # hidden affine layer
    # W2 size of H * W * num_filters * hidden_dim assume we always use pading that can keep
    # same size as input after conv layer
    # b2 size of hidden_dim
    self.params['W2'] = np.random.normal(
      0, weight_scale, (H/2 * W/2 * num_filters, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)

    # output affine layer
    # W3 size of hidden_dim * num_classes
    # b3 size of num_classes

    self.params['W22'] = np.random.normal(
      0, weight_scale, (hidden_dim, hidden_dim))
    self.params['b22'] = np.zeros(hidden_dim)

    self.params['W3'] = np.random.normal(
      0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, reward=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W22, b22 = self.params['W22'], self.params['b22']
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    # conv - relu - 2x2 max pool - affine - relu - affine - softmax

    conv_relu_pool_out, conv_relu_pool_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    affine1_relu_out, affine1_relu_cache = affine_relu_forward(conv_relu_pool_out, W2, b2)
    scores, affine2_cache = affine_forward(affine1_relu_out, W3, b3)

    if reward is None:
      return scores
    
    loss, grads = 0, {}
 
    # loss, dsoftmax = softmax_loss(scores, y)
    # add L2 regularization
    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

    daffine2 = affine_backward(reward, affine2_cache)
    daffine1_relu = affine_relu_backward(daffine2[0], affine1_relu_cache)
    dconv_relu_pool = conv_relu_pool_backward(daffine1_relu[0], conv_relu_pool_cache)



    grads['W1'] = dconv_relu_pool[1] + 0.5 * self.reg * 2 * W1
    grads['b1'] = dconv_relu_pool[2]
    grads['W2'] = daffine1_relu[1] + 0.5 * self.reg * 2 * W2
    grads['b2'] = daffine1_relu[2]
    grads['W3'] = daffine2[1] + 0.5 * self.reg * 2 * W3
    grads['b3'] = daffine2[2]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

