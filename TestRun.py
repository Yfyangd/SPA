from SPA_Net import SPANet
from CosineAnnealing import CosineAnnealingScheduler
import numpy as np
import math
import matplotlib.pyplot as plt

# Import DataSets
from keras.datasets import cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images.astype('float32');test_images.astype('float32')
train_images, test_images = train_images / 255.0, test_images / 255.0


def get_model():
    return SPANet(10)

input_shape = (None, 32, 32, 3)
model = get_model()
model.build(input_shape)
callbacks = [CosineAnnealingScheduler(T_max=100, eta_max=1e-2, eta_min=1e-4)]
model.compile(optimizer='Nadam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20, validation_split=0.25,callbacks=callbacks)


plt.style.use('seaborn')
plt.figure(figsize = (16,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training Result',fontsize=20)
plt.ylabel('Loss',fontsize=16)
plt.xlabel('Epoch',fontsize=16)
plt.legend(['accuracy','Validation_accuracy'], loc='lower right',fontsize=16)
plt.savefig('Training_Result.png')

score = model.evaluate(test_images, test_labels, verbose=0)
print('accuracy: ',score[1])
print('loss: ',score[0])