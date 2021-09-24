from tensorflow import keras
from model.utils import IdentityBlock

class Resnet50(keras.models.Model):
  def __init__(self, num_classes, **kwargs):
    super(Resnet50, self).__init__(**kwargs)

    self.conv = keras.layers.Conv2D(64, (7,7), padding='same')
    self.btch = keras.layers.BatchNormalization()
    self.maxpool = keras.layers.MaxPool2D((3,3))

    self.conv_block_1 = IdentityBlock(64, 3, name='idblock-1', enhanced=True)
    self.id_block_1 = [IdentityBlock(64, 3, name='idblock-2'),IdentityBlock(64, 3, name='idblock-3')]

    self.conv_block_2 = IdentityBlock(64, 3, name='idblock-4', enhanced=True)
    self.id_block_2 = [
        IdentityBlock(64, 3, name='idblock-5'),IdentityBlock(64, 3, name='idblock-6'),
        IdentityBlock(64, 3, name='idblock-7')
    ]

    self.conv_block_3 = IdentityBlock(64, 3, name='idblock-8', enhanced=True)
    self.id_block_3 = [
        IdentityBlock(64, 3, name='idblock-9'),IdentityBlock(64, 3, name='idblock-10'),
        IdentityBlock(64, 3, name='idblock-11'),IdentityBlock(64, 3, name='idblock-12'),
        IdentityBlock(64, 3, name='idblock-13')
    ]

    self.conv_block_4 = IdentityBlock(64, 3, name='idblock-14', enhanced=True)
    self.id_block_4 = [
        IdentityBlock(64, 3, name='idblock-15'),
        IdentityBlock(64, 3, name='idblock-16')
    ]

    self.gavg_pool = keras.layers.GlobalAveragePooling2D()
    self.classifier = keras.layers.Dense(units=num_classes, activation='softmax')

  def call(self, inputs):
    x = self.conv(inputs)
    x = self.btch(x)
    x = self.maxpool(x)

    x = self.conv_block_1(x)
    for id_block in self.id_block_1:
        x = id_block(x)

    x = self.conv_block_2(x)
    for id_block in self.id_block_2:
        x = id_block(x)

    x = self.conv_block_3(x)
    for id_block in self.id_block_3:
        x = id_block(x)

    x = self.conv_block_4(x)
    for id_block in self.id_block_4:
        x = id_block(x)
    
    x = self.gavg_pool(x)
    return self.classifier(x)