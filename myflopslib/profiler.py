import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers 
from myflopslib.myfunction import dense, conv, conv_transpose, batch_norm, zero, separable_conv, depthwise_conv
from prettytable import PrettyTable


default_header = ["Name", "Input Shape", "Output Shape", "FLOPs"]

mydict = {
    "Dense":dense,
    "Conv2D":conv,
    "Conv2DTranspose":conv_transpose,
    "BatchNormalization": batch_norm,
    "SeparableConv2D": separable_conv,
    "DepthwiseConv2D": depthwise_conv,
    "ReLU": zero
}


class Profiler:

    def __init__(self):
        self.flops = 0
        self.table = None


    def counter(self, mod: models.Model, flag = True):
        if not isinstance(mod, models.Model):
            print("Input Error")
            return 0
        for layer in mod.layers[1:]:
            key = layer.__class__.__name__
            if key in mydict.keys():
                ops = mydict[key](layer)
                input_size = "none"
                output_size = "none"
                if(hasattr(layer, "input_shape")):
                    input_size = layer.input_shape
                if(hasattr(layer, "output_shape")):
                    output_size = layer.output_shape
                self.table.add_row([key, input_size, output_size, ops])
                self.flops += ops
            elif hasattr(layer, "layers"):
                self.counter(layer, False)
            else:
                self.table.add_row([key,"Not Implemented","Not Implemented","Not Implemented"])
 
        if bool(flag):
            print(self.table)
            print("Total Cost  :  " + str(self.flops)+" FLOPs\n")
        
        return 0
    
    def compute_flops(self, mod: models.Model):
        self.flops = 0
        self.table = PrettyTable(default_header)
        self.counter(mod, 1)
        return self.flops