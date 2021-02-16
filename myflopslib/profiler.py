import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers 
from prettytable import PrettyTable
from myflopslib.myfunction import dense, conv, conv_transpose, batch_norm, zero, separable_conv, depthwise_conv, compute_act
import matplotlib.pyplot as plt


default_header = ["Layer", "Name", "Mean", "Std", "FLOPs"]

mydict = {
    "Dense":dense,
    "Conv2D":conv,
    "Reshape":zero,
    "InputLayer":zero,
    "Conv2DTranspose":conv_transpose,
    "BatchNormalization": batch_norm,
    "SeparableConv2D": separable_conv,
    "DepthwiseConv2D": depthwise_conv,
    "Activation": compute_act,
    "ReLU": zero
}


class Profiler:

    def __init__(self):
        self.flops = 0
        self.fl = []
        self.mn = []
        self.lay_name = []
        self.table = None

    

    def counter(self, mod: models.Model, flag = True):
        if not isinstance(mod, models.Model):
            print("Input Error")
            return 0
        for layer in mod._nodes_by_depth.values():
            for j in layer:
                val = j.outbound_layer
                key = val.__class__.__name__
                if key in mydict.keys():
                    ops, mean, std = mydict[key](val)
                    self.table.add_row([key, val.name, mean, std, ops])
                    if mean != 0 and ops !=0:
                      self.fl.append(ops)
                      self.mn.append(mean)
                      self.lay_name.append(key)
                    self.flops += ops
                elif hasattr(val, "layers"):
                    self.counter(val,False)
                else:
                    self.table.add_row([key, val.name, "Not Implemented","Not Implemented","Not Implemented"])

        if bool(flag):
            print(self.table)
            print("Total Cost  : {}  FLOPs\n".format(self.flops))
            plt.plot(self.mn, self.fl, 'ro')
        
        return 0
    
    def get_graphics(self, s: str):
      a = []
      b = []
      for i in zip(self.lay_name, self.mn, self.fl):
        if i[0]==s:
          a.append(i[1])
          b.append(i[2])
      plt.plot(a, b, 'ro')
    
    def compute_flops(self, mod: models.Model):
        self.flops = 0
        self.table = PrettyTable(default_header)
        self.counter(mod, 1)
        return self.flops