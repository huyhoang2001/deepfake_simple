import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import math

# Thiết lập các tham số
base_dir = os.path.abspath('dataset_1') # đường dẫn tới folder chứ dữ liệu
image_size = (224, 224) # kích thước size hình ảnh
batch_size = 32 #chỉ số so sánh 1 hình ảnh với 32 hình.
epochs = 50 # Tăng số lượng epochs. chỉ số học sâu, tăng độ chính xác.

# Tạo ImageDataGenerator với nhiều kỹ thuật Data Augmentation hơn
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Tạo các generators
train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'Train'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'Train'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, 'Test'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Kiểm tra số lượng mẫu trong test_generator
if test_generator.samples == 0:
    print("No test samples found. Please check the dataset directory.")
else:
    print(f"Number of test samples: {test_generator.samples}")

# Xây dựng mô hình với Batch Normalization và nhiều lớp hơn
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Biên dịch mô hình với optimizer Adam và Learning Rate thấp hơn
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks: Learning Rate Scheduler và Early Stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    steps_per_epoch=math.ceil(train_generator.samples / batch_size),
    validation_data=validation_generator,
    validation_steps=math.ceil(validation_generator.samples / batch_size),
    epochs=epochs,
    callbacks=[reduce_lr, early_stopping]
)

# Đánh giá mô hình trên tập test
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy:.4f}")

# Lưu mô hình
model.save('fruit_detection_model.h5')

print("Model training completed and saved.")

# Vẽ đồ thị accuracy và loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
