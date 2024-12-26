import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Параметры
DATASET_PATH = 'C:/Users/user/PycharmProjects/Stereo/app/models/model1/dataset'
TRAIN_CSV = os.path.join(DATASET_PATH, 'Metadata_Train.csv')
TEST_CSV = os.path.join(DATASET_PATH, 'Metadata_Test.csv')

# Проверка существования файлов
if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"Файл не найден: {TRAIN_CSV}")
if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(f"Файл не найден: {TEST_CSV}")

EPOCHS = 15
BATCH_SIZE = 32
SAVE_MODEL_PATH = 'saved_model.h5'
ACCURACY_LOG_PATH = 'accuracy_log.txt'
CONFUSION_MATRIX_PATH = 'confusion_matrix.png'
F1_SCORE_PATH = 'f1_score.txt'

# Функция для загрузки аудиофайлов и их меток
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    X = []
    y = []
    file_count = 0  # Счетчик загруженных файлов

    for index, row in df.iterrows():
        # Нормализуем путь к файлу
        file_path = os.path.normpath(os.path.join(DATASET_PATH, 'Train_submission', row.iloc[0]))  # Используем iloc
        print(f"Loading file: {file_path}")  # Отладочный вывод для проверки пути
        try:
            audio, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfccs = np.mean(mfccs.T, axis=0)  # Среднее значение MFCC по времени
            X.append(mfccs)
            y.append(row.iloc[1])  # Используем iloc
            file_count += 1  # Увеличиваем счетчик на 1
        except FileNotFoundError as e:
            print(f"Error loading {file_path}: {e}")
        except Exception as e:
            print(f"Unexpected error loading {file_path}: {e}")

    return np.array(X), np.array(y), file_count  # Возвращаем также количество файлов

# Загрузка данных
X_train, y_train, train_file_count = load_data(TRAIN_CSV)
X_test, y_test, test_file_count = load_data(TEST_CSV)

# Преобразование меток в категории
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Изменение формы данных для CNN
X_train = X_train.reshape(-1, 13, 1, 1)  # 13 MFCC, 1 канал
X_test = X_test.reshape(-1, 13, 1, 1)

# Создание модели CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 1), activation='relu', input_shape=(13, 1, 1)))
model.add(layers.MaxPooling2D((2, 1)))
model.add(layers.Conv2D(64, (3, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(np.unique(y_train_encoded)), activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train_encoded, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test_encoded))

# Сохранение модели
model.save(SAVE_MODEL_PATH)

# Сохранение точности в файл
with open(ACCURACY_LOG_PATH, 'w') as f:
    for epoch in range(EPOCHS):
        f.write(
            f"Epoch {epoch + 1}: Train Accuracy: {history.history['accuracy'][epoch]}, Validation Accuracy: {history.history['val_accuracy'][epoch]}\n")

# Прогнозирование на тестовых данных
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Вычисление F1 метрики
f1 = f1_score(y_test_encoded, y_pred_classes, average='weighted')

# Сохранение F1 метрики в файл
with open(F1_SCORE_PATH, 'w') as f:
    f.write(f"F1 Score: {f1}\n")

# Создание матрицы ошибок
conf_matrix = confusion_matrix(y_test_encoded, y_pred_classes)

# Визуализация матрицы ошибок
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig(CONFUSION_MATRIX_PATH)
plt.close()

# Вывод количества загруженных файлов
print(f"Обучение завершено. Использовано файлов для обучения: {train_file_count}.")
print(f"Использовано файлов для тестирования: {test_file_count}.")
print("Модель, точность, F1 метрика и матрица ошибок сохранены.")
