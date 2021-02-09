import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers 
from myflopslib.myfunction import dense, conv, conv_transpose, batch_norm, zero, separable_conv, depthwise_conv
from prettytable import PrettyTable


default_header = ["Layer", "Name", "Input Shape", "Output Shape", "FLOPs"]

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
        for layer in mod._nodes_by_depth.values():
            for j in layer:
                val = j.outbound_layer
                key = val.__class__.__name__
                if key in mydict.keys():
                    ops = mydict[key](val)
                    self.table.add_row([key, val.name, val.input_shape, val.output_shape, ops])
                    self.flops += ops
                elif hasattr(val, "layers"):
                    self.counter(val,False)
                else:
                    self.table.add_row([key, val.name, "Not Implemented","Not Implemented","Not Implemented"])
        if bool(flag):
            print(self.table)
            print("Total Cost  :  " + str(self.flops)+" FLOPs\n")
        
        return 0
    
    def compute_flops(self, mod: models.Model):
        self.flops = 0
        self.table = PrettyTable(default_header)
        self.counter(mod, 1)
        return self.flops
