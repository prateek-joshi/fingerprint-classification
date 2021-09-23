from tensorflow import keras

class IdentityBlock(keras.models.Model):
  def __init__(self, filters, kernel_size, name='', enhanced=False,**kwargs):
    super(IdentityBlock, self).__init__(name=name, **kwargs)
    self.is_enhanced = enhanced

    self.conv1 = keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same')
    self.btch1 = keras.layers.BatchNormalization()

    self.conv2 = keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same')
    self.btch2 = keras.layers.BatchNormalization()

    self.act = keras.layers.ReLU()
    self.add = keras.layers.Add()

    if self.is_enhanced:
      self.enh_conv = keras.layers.Conv2D(filters, kernel_size=(1,1), padding='same')

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.btch1(x)
    x = self.act(x)

    x = self.conv2(x)
    x = self.btch2(x)

    if self.is_enhanced:
      aux_op = self.enh_conv(inputs)
      aux_op = self.add([inputs, x])
    else:
      aux_op = self.add([inputs, x])

    return self.act(aux_op)