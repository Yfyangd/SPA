from tensorflow.keras.layers import LayerNormalization, Conv2D, Layer, Activation

class Self_Attention_Block(Layer):

    def __init__(self, filters, ratio):
        super(Self_Attention_Block, self).__init__()
        self.conv0 = Conv2D(1, (1, 1), strides=(1, 1), padding='same',
                           use_bias=False, activation=None)        
        self.softmax = Activation('softmax')
        self.conv1 = Conv2D(int(filters / ratio), (1, 1), strides=(1, 1), padding='same',
                           use_bias=False, activation=None)
        self.LN = LayerNormalization()
        self.conv2 = Conv2D(int(filters), (1, 1), strides=(1, 1), padding='same',
                           use_bias=False, activation=None)
        self.relu = Activation('relu')
        self.hard_sigmoid = Activation('hard_sigmoid')

    def call(self, inputs):
        x = self.conv0(inputs)
        self_attention = self.softmax(x)
        x = x * self_attention
        x = self.relu(self.LN(self.conv1(x)))
        excitation = self.hard_sigmoid(self.conv2(x))
        x = inputs * excitation
        return x

