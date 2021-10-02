from model.resnet import Resnet18
from tensorflow import keras
import os

IMAGES_PATH = os.path.join('data','fingerprints')
NUM_CLASSES = len(os.listdir(IMAGES_PATH))
EPOCHS = 10
BATCH_SIZE = 32
CHECKPOINT_PATH = os.path.join('model','checkpoints')
IMAGE_FILE = os.path.join('model','model_img.png')

# Create checkpoint path
if not os.path.exists(CHECKPOINT_PATH):
    os.mkdir(CHECKPOINT_PATH)

# Image data generator
datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    directory=IMAGES_PATH,
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=42
)

validation_generator = datagen.flow_from_directory(
    directory=IMAGES_PATH,
    target_size=(128, 128),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='validation',
    shuffle=True,
    seed=42
)

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

model = Resnet18(num_classes=NUM_CLASSES)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['acc']
)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_freq=BATCH_SIZE,
    save_best_only=True)

# Plot model architecture and save as image
# keras.utils.plot_model(model, to_file=IMAGE_FILE, show_shapes=True, show_layer_names=True)

print('Training started.')

model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_steps=STEP_SIZE_VALID,
    epochs=EPOCHS,
    callbacks=[model_checkpoint_callback]
)