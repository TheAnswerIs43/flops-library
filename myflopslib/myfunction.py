import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time

def default(x:layers.Layer):
  return 0


activation_function = {
    "linear": default,
    "sigmoid": default,
    "relu": default
}

def compute_time(x:layers.Layer, n:int):
  print("computing time...")
  if n==1:
    x_test = np.ones((100, x.input_shape[1]))
  elif n==3:
    x_test = np.ones((100, x.input_shape[1], x.input_shape[2], x.input_shape[3]))
  arg = tf.convert_to_tensor(x_test, dtype=tf.float32)
  steps = 50000
  tt = np.zeros(steps)
  start_time = time.time()
  for i in range(steps):
        x.call(arg)
        end_time = time.time()
        tt[i] = end_time - start_time
        start_time = end_time
  return np.mean(tt), np.std(tt)

def compute_act(x:layers.Layer):
  if hasattr(x, "activation"):
    key = x.activation.__name__
    if key in activation_function.keys():
      flops = activation_function[key](x)
      return flops, 0, 0
  return 0, 0, 0

def dense(x:layers.Layer):

  #time
  mean, std = compute_time(x, 1)
  #flops
  inp = np.prod(x.input_shape[1:])
  out = np.prod(x.output_shape[1:])
  flops = inp*out
  flops += compute_act(x)[0]
  return 2*flops, mean, std
    
def zero(x:layers.Layer):
  #flops
  flops = 0
  flops += compute_act(x)[0]
  return flops, 0, 0
        
def conv(x:layers.Layer):
  #time
  mean, std = compute_time(x, 3)
  #flops
  out = np.prod(x.output_shape[1:])
  kernel_size = np.prod(x.kernel_size)
  flops = out * (x.input_shape[-1] * kernel_size)
  flops += compute_act(x)[0]
  return 2*flops, mean, std

def conv_transpose(x:layers.Layer):
  #time
  mean, std = compute_time(x, 3)
  #flops
  out = np.prod(x.input_shape[1:])
  kernel_size = np.prod(x.kernel_size)
  flops = out * (x.output_shape[-1] * kernel_size)
  flops += compute_act(x)[0]
  return 2*flops, mean, std

def batch_norm(x:layers.Layer):
  #time
  mean, std = compute_time(x, 3)
  #flops
  flops = np.prod(x.input_shape[1:])
  return 4*flops, mean, std

def separable_conv(x:layers.Layer):
  #time
  mean, std = compute_time(x, 3)
  #flops
  first_out_channel = x.input_shape[-1] * x.depth_multiplier
  single_channel = x.output_shape[1] * x.output_shape[2]
  first_out = first_out_channel * single_channel
  first_kernel_size = np.prod(x.kernel_size)
  first_flops = first_out * first_kernel_size
  out = np.prod(x.output_shape[1:])
  kernel_size = x.input_shape[-1]*x.depth_multiplier
  flops = out * kernel_size
  return 2*(first_flops + flops), mean, std


def depthwise_conv(x:layers.Layer):
  #time
  mean, std = compute_time(x, 3)
  #flops
  out = np.prod(x.output_shape[1:])
  kernel_size = np.prod(x.kernel_size)
  flops = out * kernel_size
  return 2*flops, mean, std