import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


# class ZFConvNet1(object):
#   """
#   A deep convolutional network with the following architecture:
#   ((conv - relu) x Ns - 2x2 max pool) x Ms - (affine - relu) x Ks
#    - affine - softmax

#   The network operates on minibatches of data that have shape (N, C, H, W)
#   consisting of N images, each with height H and width W and with C input
#   channels.
#   """

#   def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
#                hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
#                dtype=np.float32, use_batchnorm=False, convnet_dim=(1, 1, 1)):
#     """
#     Initialize a new network.

#     Inputs:
#     - input_dim: Tuple (C, H, W) giving size of input data
#     - num_filters: Number of filters to use in the convolutional layer
#     - filter_size: Size of filters to use in the convolutional layer
#     - hidden_dim: Number of units to use in the fully-connected hidden layer
#     - num_classes: Number of scores to produce from the final affine layer.
#     - weight_scale: Scalar giving standard deviation for random initialization
#       of weights.
#     - reg: Scalar giving L2 regularization strength
#     - dtype: numpy datatype to use for computation.
#     """

#     self.params = {}
#     self.reg = reg
#     self.dtype = dtype
#     self.filter_size = filter_size
#     self.use_batchnorm = use_batchnorm
#     self.convnet_dim = convnet_dim
#     #  Initialize weights and biases for the three-layer convolutional


#     num_channels, H, W = input_dim
#     Ns, Ms, Ks = self.convnet_dim

#     # Ns * convoltional layer
#     # Conv_W size of (num_filters, channels, filter_size, filter_size)
#     # Conv_b size of num_filters,
#     for m in xrange(Ms):
#       m_index = m + 1
#       for n in xrange(Ns):
#         n_index = n + 1
#         self.params['Conv_W' + str(n_index) + str(m_index)] = np.random.normal(
#           0, weight_scale, (num_filters, num_channels, filter_size, filter_size))
#         self.params['Conv_b' + str(n_index) + str(m_index)] = np.zeros(num_filters)
#         if self.use_batchnorm:
#           self.params['spatial_batch_gamma' + str(n_index) + str(m_index)] = np.random.randn(num_channels)
#           self.params['spatial_batch_beta' + str(n_index) + str(m_index)] = np.random.randn(num_channels)
#           # With spital batch normalization we need to keep track of running means and
#           # variances, so we need to pass a special spital_bn_param object to each batch
#           # normalization layer.
#           self.spital_bn_params[str(n_index) + str(m_index)] = [{'mode': 'train'}]
#         num_channels = num_filters  # number of channels changed into number of filters after conv


#     # hidden affine relu layer
#     # aff_re_W size of H * W * num_filters * hidden_dim (H and W are after pools)
#     # aff_re_b size of hidden_dim

#     # after every pool layer, input dimension would divided by 2
#     H /= 2**Ms
#     W /= 2**Ms

#     aff_in_dim = H * W * num_filters

#     for k in xrange(Ks):
#       k_index = k + 1
#       self.params['aff_re_W' + str(k_index)] = np.random.normal(
#         0, weight_scale, (aff_in_dim, hidden_dim))
#       self.params['aff_re_b' + str(k_index)] = np.zeros(hidden_dim)
#       aff_in_dim = hidden_dim

#       if self.use_batchnorm:
#         self.params['batch_gamma' + str(k_index)] = np.random.randn(hidden_dim)
#         self.params['batch_beta' + str(k_index)] = np.random.randn(hidden_dim)
#         self.bn_params[str(k_index)] = [{'mode': 'train'}]
#     # output affine layer
#     # W3 size of hidden_dim * num_classes
#     # b3 size of num_classes
#     self.params['output_W'] = np.random.normal(
#       0, weight_scale, (hidden_dim, num_classes))
#     self.params['output_b'] = np.zeros(num_classes)


#     for k, v in self.params.iteritems():
#       self.params[k] = v.astype(dtype)

#   def loss(self, X, y=None):
#     """
#     Evaluate loss and gradient for the three-layer convolutional network.

#     Input / output: Same API as TwoLayerNet in fc_net.py.
#     """
#     Ns, Ms, Ks = self.convnet_dim
#     mode = 'test' if y is None else 'train'

#     # pass conv_param to the forward pass for the convolutional layer
#     filter_size = self.filter_size
#     conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

#     # pass pool_param to the forward pass for the max-pooling layer
#     pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

#     scores = None
#     # Implement the forward pass for the convolutional net
#     # Ms*(Ns*(conv - relu) - 2x2 max pool) - Ks(affine - relu) - affine - softmax
#     conv_cache = {}
#     pool_cache = {}
#     conv_out = X  # X as first conv in


#     # forward pass for conv and pool layers
#     for m in xrange(Ms):
#       m_index = m + 1
#       for n in xrange(Ns):
#         n_index = n + 1
#         conv_W = self.params['Conv_W' + str(n_index) + str(m_index)]
#         conv_b = self.params['Conv_b' + str(n_index) + str(m_index)]
#         if self.use_batchnorm:
#           self.spital_bn_params[str(n_index) + str(m_index)][mode] = mode
#           conv_out, conv_spital_batch_cache[str(n_index) + str(m_index)] = conv_spital_batchnorm_relu_forward(conv_out, conv_W, conv_b, conv_param,
#            self.params['spatial_batch_gamma' + str(n_index) + str(m_index)],
#            self.params['spatial_batch_beta' + str(n_index) + str(m_index)],
#            self.spital_bn_params[str(n_index) + str(m_index)])
#         else:
#           conv_out, conv_cache[str(n_index) + str(m_index)] = conv_relu_forward(conv_out, conv_W, conv_b, conv_param)
#       conv_out, pool_cache[str(m_index)] = max_pool_forward_fast(conv_out, pool_param)

#     # forward pass for fc affine-relu layers
#     aff_re_cache = {}
#     aff_re_out = conv_out  # conv_out from above as first input of aff_re
#     for k in xrange(Ks):
#       k_index = k + 1
#       aff_re_W = self.params['aff_re_W' + str(k_index)]
#       aff_re_b = self.params['aff_re_b' + str(k_index)]
#       aff_re_out, aff_re_cache[str(k_index)] = affine_relu_forward(aff_re_out, aff_re_W, aff_re_b)

#     # forward pass for score output layer
#     scores, out_cache = affine_forward(aff_re_out, self.params['output_W'], self.params['output_b'])


#     if y is None:
#       return scores

#     loss, grads = 0, {}
#      # Implement the backward pass for the three-layer convolutional net

#     loss, dsoftmax = softmax_loss(scores, y)
#     # add L2 regularization
#     loss += 0.5 * self.reg * np.sum(self.params['output_W']**2)
#     for m in xrange(Ms):
#       m_index = m + 1
#       for n in xrange(Ns):
#         n_index = n + 1
#         loss += 0.5 * self.reg * np.sum(self.params['Conv_W' + str(n_index) + str(m_index)]**2)
#     for k in xrange(Ks):
#       k_index = k + 1
#       loss += 0.5 * self.reg * np.sum(self.params['aff_re_W' + str(k_index)]**2)

#     # grads for last output layer
#     dout = affine_backward(dsoftmax, out_cache)
#     grads['output_W'] = dout[1] + 0.5 * self.reg * 2 * self.params['output_W']
#     grads['output_b'] = dout[2]

#     # grads for fc affine-relu layers
#     daff_re = {}
#     dupper = dout[0]
#     for k in xrange(Ks):
#       k_index = Ks - k
#       daff_re[str(k_index)] = affine_relu_backward(dupper, aff_re_cache[str(k_index)])
#       dupper = daff_re[str(k_index)][0]
#       grads['aff_re_W' + str(k_index)] = \
#         daff_re[str(k_index)][1] + 0.5 * self.reg * 2 * self.params['aff_re_W' + str(k_index)]
#       grads['aff_re_b' + str(k_index)] = daff_re[str(k_index)][2]
#     # grads for conv and pool layers
#     dpool = {}
#     dconv = {}
#     for m in xrange(Ms):
#       m_index = Ms - m
#       dpool[str(m_index)] = max_pool_backward_fast(dupper, pool_cache[str(m_index)])
#       dupper = dpool[str(m_index)]

#       for n in xrange(Ns):
#         n_index = Ns - n
#         tempa = conv_cache[str(n_index) + str(m_index)]
#         dconv[str(n_index) + str(m_index)] = conv_relu_backward(dupper, conv_cache[str(n_index) + str(m_index)])
#         dupper = dconv[str(n_index) + str(m_index)][0]
#         grads['Conv_W' + str(n_index) + str(m_index)] = \
#           dconv[str(n_index) + str(m_index)][1] + 0.5 * \
#           self.reg * self.params['Conv_W' + str(n_index) + str(m_index)]
#         grads['Conv_b' + str(n_index) + str(m_index)] = dconv[str(n_index) + str(m_index)][2]


#     return loss, grads
class ZFConvnet(object):
  """
  A  convolutional network with the following architecture:
  
  ((conv - relu)* 2 -  2x2 max pool)2 - (affine - relu) - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, use_batchnorm=False, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float64):
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
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.filter_size = filter_size

    C, H, W = input_dim
    F = num_filters
    # convoltional layer
    # W1 size of (num_filters, channels, filter_size, filter_size) 
    # assume we always use pading that can keep
    # W2 size of (num_filters, num_filters, filter_size, filter_size)
    # b1 size of num_filters
    # b2 size of num_filters

    # first pool layer
    self.params['Conv_W1_pool1'] = np.random.normal(
      0, weight_scale, (num_filters, C, filter_size, filter_size))
    self.params['Conv_b1_pool1'] = np.zeros(num_filters)

    self.params['Conv_W2_pool1'] = np.random.normal(
      0, weight_scale, (num_filters, num_filters, filter_size, filter_size))
    self.params['Conv_b2_pool1'] = np.zeros(num_filters)


    # # second pool layer
    self.params['Conv_W1_pool2'] = np.random.normal(
      0, weight_scale, (num_filters, num_filters, filter_size, filter_size))
    self.params['Conv_b1_pool2'] = np.zeros(num_filters)

    self.params['Conv_W2_pool2'] = np.random.normal(
      0, weight_scale, (num_filters, num_filters, filter_size, filter_size))
    self.params['Conv_b2_pool2'] = np.zeros(num_filters)    

    # firsr hidden affine layer
    # W size of H/4 * W/4 * num_filters * hidden_dim assume we always use pading that can keep
    # same size as input after conv layer, after two 2x2 pool layers, size become H/4 x W/4
    # b size of hidden_dim
    conv_out_dim = H/4 * W/4 * num_filters
    self.params['affine_W1'] = np.random.normal(
      0, weight_scale, (conv_out_dim, hidden_dim))
    self.params['affine_b1'] = np.zeros(hidden_dim)


    # second hidden affine layer
    # W size of hidden_dim * hidden_dim
    # self.params['affine_W2'] = np.random.normal(
    #   0, weight_scale, (hidden_dim, hidden_dim))
    # self.params['affine_b2'] = np.zeros(hidden_dim)


    # output affine layer
    # W size of hidden_dim * num_classes
    # b size of num_classes
    self.params['out_W'] = np.random.normal(
      0, weight_scale, (hidden_dim, num_classes))
    self.params['out_b'] = np.zeros(num_classes)

    # if using batchnorm, initial batchnorm params
    if self.use_batchnorm:
      self.params['Input_gamma'] = np.ones(C)
      self.params['Input_beta'] = np.zeros(C)

      self.params['Conv_gamma1_pool1'] = np.ones(F)
      self.params['Conv_beta1_pool1'] = np.zeros(F)

      self.params['Conv_gamma2_pool1'] = np.ones(F)
      self.params['Conv_beta2_pool1'] = np.zeros(F)

      self.params['Conv_gamma1_pool2'] = np.ones(F)
      self.params['Conv_beta1_pool2'] = np.zeros(F)

      self.params['Conv_gamma2_pool2'] = np.ones(F)
      self.params['Conv_beta2_pool2'] = np.zeros(F)

      self.params['affine_gamma1'] = np.ones(hidden_dim)
      self.params['affine_beta1'] = np.zeros(hidden_dim)

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    layer_number = 6 # number of batch_norm layers is 4    
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(layer_number)] 

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'

    # unpack params
    Conv_W1_pool1, Conv_b1_pool1 = self.params['Conv_W1_pool1'], self.params['Conv_b1_pool1']
    Conv_W2_pool1, Conv_b2_pool1 = self.params['Conv_W2_pool1'], self.params['Conv_b2_pool1']
    Conv_W1_pool2, Conv_b1_pool2 = self.params['Conv_W1_pool2'], self.params['Conv_b1_pool2']
    Conv_W2_pool2, Conv_b2_pool2 = self.params['Conv_W2_pool2'], self.params['Conv_b2_pool2']
    affine_W1, affine_b1 = self.params['affine_W1'], self.params['affine_b1']
    # affine_W2, affine_b2 = self.params['affine_W2'], self.params['affine_b2']
    out_W, out_b = self.params['out_W'], self.params['out_b']

    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode
      Input_gamma, Input_beta = self.params['Input_gamma'], self.params['Input_beta']  

      Conv_gamma1_pool1, Conv_beta1_pool1 = \
      self.params['Conv_gamma1_pool1'],self.params['Conv_beta1_pool1']
      Conv_gamma2_pool1, Conv_beta2_pool1 = \
      self.params['Conv_gamma2_pool1'], self.params['Conv_beta2_pool1']
      Conv_gamma1_pool2, Conv_beta1_pool2 = \
      self.params['Conv_gamma1_pool2'],self.params['Conv_beta1_pool2']
      Conv_gamma2_pool2, Conv_beta2_pool2 = \
      self.params['Conv_gamma2_pool2'], self.params['Conv_beta2_pool2']

      affine_gamma1, affine_beta1 = self.params['affine_gamma1'], self.params['affine_beta1']


    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = self.filter_size
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2} # always keep same size

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

  # architecture ((conv - relu)* 2 -  2x2 max pool)2 - (affine - relu) - affine - softmax
    if self.use_batchnorm:
      # input batch norm
      X_batch, X_batch_cache = spatial_batchnorm_forward(X, Input_gamma, Input_beta, self.bn_params[0])

      # first (conv - relu) * 2 - pool
      conv_relu_1_1_batch_out, conv_relu_1_1_batch_cache = \
      conv_spital_batchnorm_relu_forward(X_batch, Conv_W1_pool1, Conv_b1_pool1, conv_param, 
        Conv_gamma1_pool1, Conv_beta1_pool1, self.bn_params[1])

      conv_relu_1_2_batch_out, conv_relu_1_2_batch_cache = \
      conv_spital_batchnorm_relu_forward(conv_relu_1_1_batch_out, Conv_W2_pool1, Conv_b2_pool1, conv_param, 
        Conv_gamma2_pool1, Conv_beta2_pool1, self.bn_params[2])   

      pool1_out, pool1_cache = max_pool_forward_fast(conv_relu_1_2_batch_out, pool_param)

      # second (conv - relu) * 2 - pool
      conv_relu_2_1_batch_out, conv_relu_2_1_batch_cache = \
      conv_spital_batchnorm_relu_forward(pool1_out, Conv_W1_pool2, Conv_b1_pool2, conv_param, 
        Conv_gamma1_pool2, Conv_beta1_pool2, self.bn_params[3])

      conv_relu_2_2_batch_out, conv_relu_2_2_batch_cache = \
      conv_spital_batchnorm_relu_forward(conv_relu_2_1_batch_out, Conv_W2_pool2, Conv_b2_pool2, conv_param, 
        Conv_gamma2_pool2, Conv_beta2_pool2, self.bn_params[4])   

      pool2_out, pool2_cache = max_pool_forward_fast(conv_relu_2_2_batch_out, pool_param)

      # affine relu
      affine_relu_1_batch_out, affine_relu_1_batch_cache = \
      affine_batchnorm_relu_forward(pool2_out, affine_W1, affine_b1, affine_gamma1, affine_beta1, self.bn_params[5])

      # affine
      scores, out_cache = affine_forward(affine_relu_1_batch_out, out_W, out_b)

    else:
      conv_relu_1_1_out, conv_relu_1_1_cache = conv_relu_forward(X, Conv_W1_pool1, Conv_b1_pool1, conv_param) 
      conv_relu_1_2_out, conv_relu_1_2_cache = conv_relu_forward(conv_relu_1_1_out, Conv_W2_pool1, Conv_b2_pool1, conv_param) 
      pool1_out, pool1_cache = max_pool_forward_fast(conv_relu_1_2_out, pool_param)
      conv_relu_2_1_out, conv_relu_2_1_cache = conv_relu_forward(pool1_out, Conv_W1_pool2, Conv_b1_pool2, conv_param)
      conv_relu_2_2_out, conv_relu_2_2_cache = conv_relu_forward(conv_relu_2_1_out, Conv_W2_pool2, Conv_b2_pool2, conv_param) 
      pool2_out, pool2_cache = max_pool_forward_fast(conv_relu_2_2_out, pool_param)
      affine_relu_1_out, affine_relu_1_cache = affine_relu_forward(pool2_out, affine_W1, affine_b1)
      # affine_relu_2_out, affine_relu_2_cache = affine_relu_forward(affine_relu_1_out, affine_W2, affine_b2)
      scores, out_cache = affine_forward(affine_relu_1_out, out_W, out_b)

    
    if y is None:
      return scores
    
    loss, grads = 0, {}

    loss, dsoftmax = softmax_loss(scores, y)

    # # add L2 regularization
    # loss += 0.5 * self.reg * (np.sum(Conv_W1_pool1**2) + np.sum(Conv_W2_pool1**2) + 
    #   np.sum(Conv_W1_pool2**2) + np.sum(Conv_W2_pool2**2) + np.sum(affine_W1**2) +
    #   + np.sum(out_W**2))
    # add L2 regularization
    loss += 0.5 * self.reg * (np.sum(Conv_W1_pool1**2) + np.sum(Conv_W2_pool1**2) +\
     np.sum(Conv_W1_pool2**2) + np.sum(Conv_W2_pool2**2) + np.sum(affine_W1**2) + np.sum(out_W**2))

    if self.use_batchnorm:
      dscore = affine_backward(dsoftmax,out_cache)
      daffine_relu_batch_1 = affine_batchnorm_relu_backward(dscore[0], affine_relu_1_batch_cache)

      dpool2 = max_pool_backward_fast(daffine_relu_batch_1[0], pool2_cache)
      dconv_relu_batch_2_2 = conv_spital_batchnorm_relu_backward(dpool2, conv_relu_2_2_batch_cache)
      dconv_relu_batch_2_1 =conv_spital_batchnorm_relu_backward(dconv_relu_batch_2_2[0], conv_relu_2_1_batch_cache)

      dpool1 = max_pool_backward_fast(dconv_relu_batch_2_1[0], pool1_cache)
      dconv_relu_batch_1_2 = conv_spital_batchnorm_relu_backward(dpool1, conv_relu_1_2_batch_cache)
      dconv_relu_batch_1_1 =conv_spital_batchnorm_relu_backward(dconv_relu_batch_1_2[0], conv_relu_1_1_batch_cache)

      dX_batch = spatial_batchnorm_backward(dconv_relu_batch_1_1[0], X_batch_cache)
    else:
      dscore = affine_backward(dsoftmax,out_cache)
      # daffine_relu_2 =  affine_relu_backward(dscore[0], affine_relu_2_cache)
      daffine_relu_1 = affine_relu_backward(dscore[0], affine_relu_1_cache)
      dpool2 = max_pool_backward_fast(daffine_relu_1[0], pool2_cache)
      dconv_relu_2_2 = conv_relu_backward(dpool2, conv_relu_2_2_cache)
      dconv_relu_2_1 = conv_relu_backward(dconv_relu_2_2[0], conv_relu_2_1_cache)
      dpool1 = max_pool_backward_fast(dconv_relu_2_1[0], pool1_cache)
      dconv_relu_1_2 = conv_relu_backward(dpool1, conv_relu_1_2_cache)
      dconv_relu_1_1 =conv_relu_backward(dconv_relu_1_2[0], conv_relu_1_1_cache)

    if self.use_batchnorm:
      grads['Input_gamma'] = dX_batch[1]
      grads['Input_beta'] = dX_batch[2]
      grads['Conv_gamma1_pool1'] = dconv_relu_batch_1_1[4]
      grads['Conv_beta1_pool1'] = dconv_relu_batch_1_1[5]
      grads['Conv_gamma2_pool1'] = dconv_relu_batch_1_2[4]
      grads['Conv_beta2_pool1'] = dconv_relu_batch_1_2[5]

      grads['Conv_gamma1_pool2'] = dconv_relu_batch_2_1[4]
      grads['Conv_beta1_pool2'] = dconv_relu_batch_2_1[5]
      grads['Conv_gamma2_pool2'] = dconv_relu_batch_2_2[4]
      grads['Conv_beta2_pool2'] = dconv_relu_batch_2_2[5]

      grads['affine_gamma1'] = daffine_relu_batch_1[4]
      grads['affine_beta1'] = daffine_relu_batch_1[5]

      grads['Conv_W1_pool1'] = dconv_relu_batch_1_1[1] + 0.5 * self.reg * 2 * Conv_W1_pool1
      grads['Conv_b1_pool1'] = dconv_relu_batch_1_1[2]
      grads['Conv_W2_pool1'] = dconv_relu_batch_1_2[1] + 0.5 * self.reg * 2 * Conv_W2_pool1
      grads['Conv_b2_pool1'] = dconv_relu_batch_1_2[2]
      grads['Conv_W1_pool2'] = dconv_relu_batch_2_1[1] + 0.5 * self.reg * 2 * Conv_W1_pool2
      grads['Conv_b1_pool2'] = dconv_relu_batch_2_1[2]
      grads['Conv_W2_pool2'] = dconv_relu_batch_2_2[1] + 0.5 * self.reg * 2 * Conv_W2_pool2
      grads['Conv_b2_pool2'] = dconv_relu_batch_2_2[2]
      grads['affine_W1'] = daffine_relu_batch_1[1] +  0.5 * self.reg * 2 * affine_W1
      grads['affine_b1'] = daffine_relu_batch_1[2]
      # grads['affine_W2'] = daffine_relu_2[1] +  0.5 * self.reg * 2 * affine_W2
      # grads['affine_b2'] = daffine_relu_2[2]
      grads['out_W'] = dscore[1] +  0.5 * self.reg * 2 * out_W
      grads['out_b'] = dscore[2]

    else:
      grads['Conv_W1_pool1'] = dconv_relu_1_1[1] + 0.5 * self.reg * 2 * Conv_W1_pool1
      grads['Conv_b1_pool1'] = dconv_relu_1_1[2]
      grads['Conv_W2_pool1'] = dconv_relu_1_2[1] + 0.5 * self.reg * 2 * Conv_W2_pool1
      grads['Conv_b2_pool1'] = dconv_relu_1_2[2]
      grads['Conv_W1_pool2'] = dconv_relu_2_1[1] + 0.5 * self.reg * 2 * Conv_W1_pool2
      grads['Conv_b1_pool2'] = dconv_relu_2_1[2]
      grads['Conv_W2_pool2'] = dconv_relu_2_2[1] + 0.5 * self.reg * 2 * Conv_W2_pool2
      grads['Conv_b2_pool2'] = dconv_relu_2_2[2]
      grads['affine_W1'] = daffine_relu_1[1] +  0.5 * self.reg * 2 * affine_W1
      grads['affine_b1'] = daffine_relu_1[2]
      # grads['affine_W2'] = daffine_relu_2[1] +  0.5 * self.reg * 2 * affine_W2
      # grads['affine_b2'] = daffine_relu_2[2]
      grads['out_W'] = dscore[1] +  0.5 * self.reg * 2 * out_W
      grads['out_b'] = dscore[2]


    return loss, grads



class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
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
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
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

 
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
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
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # conv - relu - 2x2 max pool - affine - relu - affine - softmax

    conv_relu_pool_out, conv_relu_pool_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    affine1_relu_out, affine1_relu_cache = affine_relu_forward(conv_relu_pool_out, W2, b2)
    scores, affine2_cache = affine_forward(affine1_relu_out, W3, b3)
    # conv_relu_out, conv_relu_cache = conv_relu_forward(X, W1, b1, conv_param)
    # pool_out, pool_cache = max_pool_forward_fast(conv_relu_out, pool_param)
    # affine1_relu_out, affine1_relu_cache = affine_relu_forward(pool_out, W2, b2)
    # affine2_relu_out, affine2_relu_cache = affine_relu_forward(affine1_relu_out, W22, b22)
    # scores, affine2_cache = affine_forward(affine2_relu_out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dsoftmax = softmax_loss(scores, y)
    # add L2 regularization
    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

    daffine2 = affine_backward(dsoftmax, affine2_cache)
    daffine1_relu = affine_relu_backward(daffine2[0], affine1_relu_cache)
    dconv_relu_pool = conv_relu_pool_backward(daffine1_relu[0], conv_relu_pool_cache)

    # dscores = affine_backward(dsoftmax, affine2_cache)
    # daffine2_relu = affine_relu_backward(dscores[0], affine2_relu_cache)
    # daffine1_relu = affine_relu_backward(daffine2_relu[0], affine1_relu_cache)
    # dpool = max_pool_backward_fast(daffine2_relu[0], pool_cache)

    # dconv_relu = conv_relu_backward(dpool, conv_relu_cache)

    grads['W1'] = dconv_relu_pool[1] + 0.5 * self.reg * 2 * W1
    grads['b1'] = dconv_relu_pool[2]

    # grads['W1'] = dconv_relu[1] + 0.5 * self.reg * 2 * W1
    # grads['b1'] = dconv_relu[2]
    grads['W2'] = daffine1_relu[1] + 0.5 * self.reg * 2 * W2
    grads['b2'] = daffine1_relu[2]
    grads['W3'] = daffine2[1] + 0.5 * self.reg * 2 * W3
    grads['b3'] = daffine2[2]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

