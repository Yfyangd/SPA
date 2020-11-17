import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, BatchNormalization, Reshape, Activation, add
from Self_Proliferate_and_Attention import Self_Proliferate_and_Attention_Block

class SPANet(tf.keras.Model):
    
    def __init__(self, classes):
        super(SPANet, self).__init__()
        self.classes = classes
        self.conv1 = Conv2D(16, (3, 3), strides=(2, 2), padding='same',
                            activation=None, use_bias=False)
        self.conv2 = Conv2D(960, (1, 1), strides=(1, 1), padding='same',
                            activation=None, use_bias=False)
        self.conv3 = Conv2D(1280, (1, 1), strides=(1, 1), padding='same',
                            activation=None, use_bias=False)
        self.conv4 = Conv2D(self.classes, (1, 1), strides=(1, 1), padding='same',
                            activation=None, use_bias=False)
        for i in range(3):
            setattr(self, f"batchnorm{i+1}", BatchNormalization())
        self.relu = Activation('relu')
        self.softmax = Activation('softmax')
        self.pooling = GlobalAveragePooling2D()

        self.dwkernels = [3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]
        self.strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
        self.exps = [16, 48, 72, 72, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960, 960, 960]
        self.outs = [16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160]
        self.ratios = [2] * 16
        self.use_sa = [True, True, True, True, True, True, True, True,
                        True, True, True, True, True, True, True, True]
        for i, args in enumerate(zip(self.dwkernels, self.strides, self.exps, self.outs, self.ratios, self.use_sa)):
            setattr(self, f"Self_Proliferate_and_Attention_Block{i}", Self_Proliferate_and_Attention_Block(*args))

    def call(self, inputs):
        x = self.relu(self.batchnorm1(self.conv1(inputs)))
        # Iterate through Ghost Bottlenecks
        for i in range(16):
            x = getattr(self, f"Self_Proliferate_and_Attention_Block{i}")(x)
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.pooling(x)
        x = Reshape((1, 1, int(x.shape[1])))(x)
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.conv4(x)
        x = tf.squeeze(x, 1)
        x = tf.squeeze(x, 1)
        output = self.softmax(x)
        return output

