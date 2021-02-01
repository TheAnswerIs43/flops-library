import numpy as np
from keras import layers


def default(x:layers.Layer):
  return 0


activation_function = {
    "linear": default,
    "sigmoid": default,
    "relu": default
}

def compute_act(x:layers.Layer):
  flops = 0
  if hasattr(x, "activation"):
    key = x.activation.__name__
    if key in activation_function.keys():
      flops += activation_function[key](x)
  return flops

def dense(x:layers.Layer):
  inp = np.prod(x.input_shape[1:])
  out = np.prod(x.output_shape[1:])
  flops = inp*out*2
  flops += compute_act(x)
  return flops
    
def zero(x:layers.Layer):
  flops = 0
  flops += compute_act(x)
  return flops
        
def conv(x:layers.Layer):
  out = np.prod(x.output_shape[1:])
  kernel_size = np.prod(x.kernel_size)
  flops = 2 * out * (x.input_shape[-1] * kernel_size)
  flops += compute_act(x)
  return flops

def batch_norm(x:layers.Layer):
  flops = np.prod(x.input_shape[1:])
  return flops*2
