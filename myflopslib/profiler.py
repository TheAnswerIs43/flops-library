import numpy as np
import keras
from keras import layers 
from myflopslib.myfunction import *
from prettytable import PrettyTable


default_header = ["Name", "Input Shape", "Output Shape", "FLOPs"]

mydict = {
    "Dense":dense,
    "Conv2D":conv,
    "Conv2DTranspose":conv,
    "BatchNormalization": batch_norm,
    "ReLu": zero
}


class Profiler:

    def __init__(self):
        self.flops = 0
        self.table = None


    def counter(self, mod: keras.models.Model, flag = True):
        if not isinstance(mod, keras.models.Model):
            print("Input Error")
            return 0
        for layer in mod.layers[1:]:
            key = layer.__class__.__name__
            if key in mydict.keys():
                ops = mydict[key](layer)
                self.table.add_row([key, layer.input_shape, layer.output_shape, ops])
                self.flops += ops
            elif hasattr(layer, "layers"):
                self.counter(layer, False)
            else:
                self.table.add_row([key,layer.input_shape,layer.output_shape,"Not Implemented"])
 
        if bool(flag):
            print(self.table)
            print("Total Cost  :  " + str(self.flops)+" FLOPs\n")
        
        return 0
    
    def compute_flops(self, mod: keras.models.Model):
        self.flops = 0
        self.table = PrettyTable(default_header)
        self.counter(mod, 1)
        return self.flops