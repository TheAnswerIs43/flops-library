import numpy as np
from tensorflow.keras import layers


def default(x:layers.Layer):
  return 0


activation_function = {
    "linear": default,
    "sigmoid": default,
    "relu": default
}

def compute_act(x:layers.Layer):
  if hasattr(x, "activation"):
    key = x.activation.__name__
    if key in activation_function.keys():
      flops = activation_function[key](x)
      return flops
  return 0

def dense(x:layers.Layer):
  inp = np.prod(x.input_shape[1:])
  out = np.prod(x.output_shape[1:])
  flops = inp*out
  flops += compute_act(x)
  return 2*flops
    
def zero(x:layers.Layer):
  flops = 0
  flops += compute_act(x)
  return flops
        
def conv(x:layers.Layer):
  out = np.prod(x.output_shape[1:])
  kernel_size = np.prod(x.kernel_size)
  flops = out * (x.input_shape[-1] * kernel_size)
  flops += compute_act(x)
  return 2*flops

def conv_transpose(x:layers.Layer):
  out = np.prod(x.input_shape[1:])
  kernel_size = np.prod(x.kernel_size)
  flops = out * (x.output_shape[-1] * kernel_size)
  flops += compute_act(x)
  return 2*flops

def batch_norm(x:layers.Layer):
  flops = np.prod(x.input_shape[1:])
  return 4*flops

def separable_conv(x:layers.Layer):
  first_out_channel = x.input_shape[-1] * x.depth_multiplier
  single_channel = x.output_shape[1] * x.output_shape[2]
  first_out = first_out_channel * single_channel
  first_kernel_size = np.prod(x.kernel_size)
  first_flops = first_out * first_kernel_size
  out = np.prod(x.output_shape[1:])
  kernel_size = x.input_shape[-1]*x.depth_multiplier
  flops = out * kernel_size
  return 2*(first_flops + flops)


def depthwise_conv(x:layers.Layer):
  out = np.prod(x.output_shape[1:])
  kernel_size = np.prod(x.kernel_size)
  flops = out * kernel_size
  return 2*flops


