from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

train_generator=datagen.flow_from_directory(
    'data/train',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical'
)

validation_generator=datagen.flow_from_directory(
    'data/validation',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical'
)

