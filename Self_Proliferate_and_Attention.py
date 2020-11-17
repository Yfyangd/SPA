import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, DepthwiseConv2D, Layer, Activation, add
from Self_Attention import Self_Attention_Block
from Self_Proliferate import Self_Proliferate_Block

class Self_Proliferate_and_Attention_Block(Layer):

    def __init__(self, dwkernel, strides, exp, out, ratio, use_se):
        super(Self_Proliferate_and_Attention_Block, self).__init__()
        self.strides = strides
        self.use_se = use_se
        self.conv = Conv2D(out, (1, 1), strides=(1, 1), padding='same',
                           activation=None, use_bias=False)
        self.relu = Activation('relu')
        self.depthconv1 = DepthwiseConv2D(dwkernel, strides, padding='same', depth_multiplier=ratio-1,
                                         activation=None, use_bias=False)
        self.depthconv2 = DepthwiseConv2D(dwkernel, strides, padding='same', depth_multiplier=ratio-1,
                                         activation=None, use_bias=False)
        for i in range(5):
            setattr(self, f"batchnorm{i+1}", BatchNormalization())
        self.SAB1 = Self_Proliferate_Block(exp, ratio, 1, 3)
        self.SAB2 = Self_Proliferate_Block(out, ratio, 1, 3)
        self.se = Self_Attention_Block(exp, ratio)

    def call(self, inputs):
        x = self.batchnorm1(self.depthconv1(inputs))
        x = self.batchnorm2(self.conv(x))

        y = self.relu(self.batchnorm3(self.SAB1(inputs)))
        if self.strides > 1:
            y = self.relu(self.batchnorm4(self.depthconv2(y)))
        if self.use_se:
            y = self.se(y)
        y = self.batchnorm5(self.SAB2(y))
        return add([x, y])

