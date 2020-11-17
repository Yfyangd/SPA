from tensorflow.keras.layers import Conv2D, Concatenate, DepthwiseConv2D, Layer, Activation
from math import ceil

class Self_Proliferate_Block(Layer):

    def __init__(self, out, ratio, convkernel, dwkernel):
        super(Self_Proliferate_Block, self).__init__()
        self.ratio = ratio
        self.out = out
        self.conv_out_channel = ceil(self.out * 1.0 / ratio)
        self.conv = Conv2D(int(self.conv_out_channel), (convkernel, convkernel), use_bias=False,
                           strides=(1, 1), padding='same', activation=None)
        self.depthconv = DepthwiseConv2D(dwkernel, 1, padding='same', use_bias=False,
                                         depth_multiplier=ratio-1, activation=None)
        self.concat = Concatenate()

    def call(self, inputs):
        x = self.conv(inputs)
        if self.ratio == 1:
            return x
        dw = self.depthconv(x)
        dw = dw[:, :, :, :int(self.out - self.conv_out_channel)]
        output = self.concat([x, dw])
        return output

