import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import save_model
import pandas as pd

# Параметры
base_dir = 'C:/Users/user/PycharmProjects/Stereo/app/models/model1/dataset1/train'
batch_size = 32
img_height, img_width = 150, 150
num_epochs = 25

# Генерация данных
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Получение имен классов
class_indices = train_generator.class_indices
class_names = list(class_indices.keys())
with open('class_labels.txt', 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

# Создание модели MobileNet
base_model = MobileNet(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Замораживаем слои MobileNet

# Создание модели
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Глобальное усреднение
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(train_generator, validation_data=validation_generator, epochs=num_epochs)

# Сохранение модели
model.save('trained_model.h5')

# Оценка модели на тестовом наборе
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, '../test'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Получение предсказаний
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Вычисление точности
accuracy = accuracy_score(true_classes, predicted_classes)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Матрица ошибок
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix2.png')
plt.show()

# Сохранение матрицы ошибок и точности
results = pd.DataFrame(cm)
results.to_csv('confusion_matrix2.csv', index=False)
with open('model_accuracy2.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy * 100:.2f}%\n')
