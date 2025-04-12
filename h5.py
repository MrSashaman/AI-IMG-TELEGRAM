import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import os
# 3. Создание модели (с использованием MobileNetV2 для переноса обучения)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# 1. Подготовка данных
data_dir = 'dataset/'
img_height = 128
img_width = 128
batch_size = 32
epochs = 50 # Увеличьте при необходимости, но следите за переобучением

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            img = Image.open(filepath).convert('RGB').resize((img_height, img_width))
            img = np.array(img) / 255.0
            if img.shape != (img_height, img_width, 3):
                print(f"Изображение {filename} имеет некорректную размерность: {img.shape}. Пропущено.")
                continue
            images.append(img)
            labels.append(folder.split(os.sep)[-1])
        except (IOError, OSError) as e:
            print(f"Не удалось открыть или обработать файл {filename}: {e}. Пропущено.")
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {e}. Пропущено.")
    return images, labels

# Загружаем данные
x = []
y = []
for class_name in ['человек', 'животное', 'предмет']:
    folder_path = os.path.join(data_dir, class_name)
    if os.path.exists(folder_path):
        images, labels = load_images_from_folder(folder_path)
        x.extend(images)
        y.extend([class_name] * len(images))
    else:
        print(f"Папка {folder_path} не найдена!")

if not x:
    print("Ошибка: Не загружено ни одного изображения! Проверьте пути к данным.")
    exit()

# Преобразуем метки в числовые значения
unique_labels = list(set(y))
label_to_index = {label: i for i, label in enumerate(unique_labels)}
y = [label_to_index[label] for label in y]

try:
    x = np.array(x)
    y = np.array(y)
except ValueError as e:
    print(f"Ошибка при создании массивов NumPy: {e}")
    exit()


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# 2. Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(x_train)  #  ВАЖНО: Вычислить параметры для нормализации на обучающей выборке

# Создайте генератор для обучения
train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)



base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False # Замораживаем базовую модель

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  #или Flatten()
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5), # Добавляем Dropout для регуляризации
    layers.Dense(len(unique_labels), activation='softmax')
])

# 4. Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Обучение модели
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Увеличен patience
history = model.fit(train_generator,
                    steps_per_epoch=len(x_train) // batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping])

# 6. Сохранение модели
model.save('model.h5')
print("Модель сохранена в model.h5")
