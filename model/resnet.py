from tensorflow import keras
from model.utils import IdentityBlock

class Resnet18(keras.models.Model):
  def __init__(self, num_classes, **kwargs):
    super(Resnet18, self).__init__(**kwargs)

    self.conv = keras.layers.Conv2D(32, (7,7), padding='same', strides=(2,2))
    # self.btch = keras.layers.BatchNormalization()
    self.maxpool = keras.layers.MaxPool2D((3,3), strides=(2,2))

    # self.conv_block_1 = IdentityBlock(32, 3, name='idblock-1', enhanced=True)
    self.id_block_1 = [IdentityBlock(32, 3, name='idblock-2'),IdentityBlock(32, 3, name='idblock-3')]

    self.conv_block_2 = IdentityBlock(32, 3, name='idblock-4', enhanced=True)
    self.id_block_2 = [
        # IdentityBlock(32, 3, name='idblock-5'),IdentityBlock(32, 3, name='idblock-6'),
        IdentityBlock(32, 3, name='idblock-7')
    ]

    self.gavg_pool = keras.layers.GlobalAveragePooling2D()
    self.fc = keras.layers.Dense(units=128, activation='relu')
    self.classifier = keras.layers.Dense(units=num_classes, activation='softmax')

  def call(self, inputs):
    x = self.conv(inputs)
    # x = self.btch(x)
    x = self.maxpool(x)

    # x = self.conv_block_1(x)
    for id_block in self.id_block_1:
        x = id_block(x)

    x = self.conv_block_2(x)
    for id_block in self.id_block_2:
        x = id_block(x)
    
    x = self.gavg_pool(x)
    x = self.fc(x)
    return self.classifier(x)