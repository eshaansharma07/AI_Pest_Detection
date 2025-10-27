
# model_training.py
# Author: Eshaan Sharma | AI Pest Detection Research ML Project
# NOTE: This script requires TensorFlow. It's provided for completeness.
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_and_train_model():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        '../data/',
        target_size=(128, 128),
        batch_size=8,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        '../data/',
        target_size=(128, 128),
        batch_size=8,
        class_mode='categorical',
        subset='validation'
    )

    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(train_data.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data, epochs=3)
    model.save('../models/pest_detector.h5')
    print('Model trained and saved to ../models/pest_detector.h5')

if __name__ == '__main__':
    build_and_train_model()
