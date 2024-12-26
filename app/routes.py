from flask import Blueprint, render_template, request, jsonify
import os
import numpy as np
import librosa
import tensorflow as tf
from PIL import Image
import torch  # Импортируем библиотеку torch для YOLOv5
import cv2  # Импортируем библиотеку OpenCV

# Создание Blueprint
main = Blueprint('main', __name__)

# Путь к моделям
audio_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/model1/saved_model.keras'))
audio_model_cnn_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/model2/cnn_model.keras'))
image_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/model1/trained_model.keras'))
image_classes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/model1/class_labels.txt'))

# Проверка наличия файлов моделей
if not os.path.exists(audio_model_path):
    raise FileNotFoundError(f"Audio model file not found: {audio_model_path}")

if not os.path.exists(image_model_path):
    raise FileNotFoundError(f"Image model file not found: {image_model_path}")

if not os.path.exists(image_classes_path):
    raise FileNotFoundError(f"Image classes file not found: {image_classes_path}")

# Загрузка моделей
audio_model = tf.keras.models.load_model(audio_model_path)
audio_model_cnn = tf.keras.models.load_model(audio_model_cnn_path)
image_model = tf.keras.models.load_model(image_model_path)

def load_image_classes(file_path):
    with open(file_path, 'r') as f:
        return {idx: line.strip() for idx, line in enumerate(f)}

# Загрузка классов
image_class_labels = load_image_classes(image_classes_path)

# Добавляем классы для YOLOv5
yolo_classes = {
    0: 'Accordion',
    1: 'Clarinet',
    2: 'Daf',
    3: 'Flute',
    4: 'Guitar',
    5: 'Kamanche',
    6: 'Piano',
    7: 'Santur',
    8: 'Setar',
    9: 'Tombak',
    10: 'Tuba',
    11: 'Violin'
}

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/about')
def about():
    return render_template('about.html')

@main.route('/dense', methods=['GET', 'POST'])
def dense():
    predictions = []
    audio_class_labels = {0: 'Guitar_Sound', 1: 'Drum_Sound', 2: 'Violin_Sound', 3: 'Piano_Sound'}
    uploaded_files = []

    if request.method == 'POST':
        model_type = request.form.get('modelType')
        files = request.files.getlist('files')

        for file in files:
            if file.filename == '':
                continue

            try:
                upload_folder = 'app/static/uploads'
                os.makedirs(upload_folder, exist_ok=True)
                file_path = os.path.join(upload_folder, file.filename)
                file.save(file_path)

                uploaded_files.append(file.filename)

                if model_type == 'audio':
                    audio, sr = librosa.load(file_path, sr=None)
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                    mfccs = np.mean(mfccs.T, axis=0).reshape(-1, 13, 1, 1)

                    prediction = audio_model.predict(mfccs)
                    predicted_class = np.argmax(prediction, axis=1)
                    predicted_labels = [audio_class_labels[idx] for idx in predicted_class]
                    predictions.append(predicted_labels)

                elif model_type == 'image':
                    img = Image.open(file_path)
                    img = img.resize((150, 150))
                    img_array = np.array(img) / 255.0
                    img_array = img_array.reshape(-1, 150, 150, 3)

                    prediction = image_model.predict(img_array)
                    top_indices = np.argsort(prediction[0])[-3:][::-1]
                    top_labels = [image_class_labels[idx] for idx in top_indices]
                    top_probabilities = prediction[0][top_indices]

                    predictions.append(list(zip(top_labels, top_probabilities)))

            except Exception as e:
                print(f"Ошибка при обработке файла {file.filename}: {e}")

    return render_template('dense.html', predictions=predictions, uploaded_files=uploaded_files)

@main.route('/cnn', methods=['GET', 'POST'])
def cnn():
    predictions = []
    audio_class_labels = {0: 'Guitar_Sound', 1: 'Drum_Sound', 2: 'Violin_Sound', 3: 'Piano_Sound'}
    uploaded_files = []

    if request.method == 'POST':
        model_type = request.form.get('modelType')
        files = request.files.getlist('files')

        for file in files:
            if file.filename == '':
                continue

            try:
                upload_folder = 'app/static/uploads'
                os.makedirs(upload_folder, exist_ok=True)
                file_path = os.path.join(upload_folder, file.filename)
                file.save(file_path)

                uploaded_files.append(file.filename)
                audio, sr = librosa.load(file_path, sr=None)
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                mfccs = np.mean(mfccs.T, axis=0).reshape(-1, 13, 1, 1)

                prediction = audio_model_cnn.predict(mfccs)
                predicted_class = np.argmax(prediction, axis=1)
                predicted_labels = [audio_class_labels[idx] for idx in predicted_class]
                predictions.append(predicted_labels)
            except Exception as e:
                print(f"Ошибка при обработке файла {file.filename}: {e}")

    return render_template('cnn.html', predictions=predictions, uploaded_files=uploaded_files)

@main.route('/yolov5', methods=['GET', 'POST'])
def yolov5():
    predictions = []
    uploaded_files = []

    if request.method == 'POST':
        files = request.files.getlist('files')
        selected_classes = request.form.getlist('classes')  # Получаем выбранные классы

        for file in files:
            if file.filename == '':
                continue

            try:
                upload_folder = 'app/static/uploads'
                os.makedirs(upload_folder, exist_ok=True)
                file_path = os.path.join(upload_folder, file.filename)
                file.save(file_path)

                uploaded_files.append(file.filename)

                # Загрузка изображения
                img = cv2.imread(file_path)

                # Загрузка кастомной модели YOLO
                model = torch.hub.load('ultralytics/yolov5', 'custom', path='app/yolov5/best.pt')

                # Выполнение инференса
                results = model(img)

                # Получаем результаты в формате DataFrame
                results_df = results.pandas().xyxy[0]  # Получаем результаты в формате DataFrame

                # Фильтруем результаты по выбранным классам, если они указаны
                if selected_classes:
                    selected_classes = [int(cls) for cls in selected_classes]  # Преобразуем в int
                    results_df = results_df[results_df['class'].isin(selected_classes)]

                # Применяем порог уверенности
                confidence_threshold = 0.3
                results_df = results_df[results_df['confidence'] >= confidence_threshold]

                # Сохранение результатов
                predictions.append({
                    'file': file.filename,
                    'results': results_df
                })

                # Отрисовка результатов на изображении
                filtered_image = np.array(img)
                for _, row in results_df.iterrows():
                    label = row['name']
                    conf = row['confidence']
                    xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
                    color = (0, 255, 0)  # Зеленый цвет для рамки
                    cv2.rectangle(filtered_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 1)
                    cv2.putText(filtered_image, f'{label} {conf:.2f}', (int(xmin), int(ymin) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Сохранение изображения с детекцией
                output_image_path = os.path.join(upload_folder, f'detected_{file.filename}')
                cv2.imwrite(output_image_path, filtered_image)  # Сохраняем изображение с детекцией

            except Exception as e:
                print(f"Ошибка при обработке файла {file.filename}: {e}")

    return render_template('yolov5.html', predictions=predictions, uploaded_files=uploaded_files, image_class_labels=yolo_classes)

# API для Dense
@main.route('/api/dense', methods=['POST'])
def api_dense():
    predictions = []
    uploaded_files = []
    audio_class_labels = {0: 'Guitar_Sound', 1: 'Drum_Sound', 2: 'Violin_Sound', 3: 'Piano_Sound'}
    files = request.files.getlist('files')

    for file in files:
        if file.filename == '':
            continue

        try:
            upload_folder = 'app/static/uploads'
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            uploaded_files.append(file.filename)

            audio, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfccs = np.mean(mfccs.T, axis=0).reshape(-1, 13, 1, 1)

            prediction = audio_model.predict(mfccs)
            predicted_class = np.argmax(prediction, axis=1)
            predicted_labels = [audio_class_labels[idx] for idx in predicted_class]
            predictions.append(predicted_labels)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"predictions": predictions, "uploaded_files": uploaded_files})

# API для CNN
@main.route('/api/cnn', methods=['POST'])
def api_cnn():
    predictions = []
    uploaded_files = []
    audio_class_labels = {0: 'Guitar_Sound', 1: 'Drum_Sound', 2: 'Violin_Sound', 3: 'Piano_Sound'}
    files = request.files.getlist('files')

    for file in files:
        if file.filename == '':
            continue

        try:
            upload_folder = 'app/static/uploads'
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            uploaded_files.append(file.filename)
            audio, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfccs = np.mean(mfccs.T, axis=0).reshape(-1, 13, 1, 1)

            prediction = audio_model_cnn.predict(mfccs)
            predicted_class = np.argmax(prediction, axis=1)
            predicted_labels = [audio_class_labels[idx] for idx in predicted_class]
            predictions.append(predicted_labels)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"predictions": predictions, "uploaded_files": uploaded_files})

# API для YOLOv5
@main.route('/api/yolov5', methods=['POST'])
def api_yolov5():
    predictions = []
    uploaded_files = []
    files = request.files.getlist('files')

    for file in files:
        if file.filename == '':
            continue

        try:
            upload_folder = 'app/static/uploads'
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            uploaded_files.append(file.filename)

            # Загрузка изображения
            img = cv2.imread(file_path)

            # Выполнение инференса
            results = torch.hub.load('ultralytics/yolov5', 'custom', path='app/yolov5/best.pt')(img)

            # Получаем результаты в формате DataFrame
            results_df = results.pandas().xyxy[0]

            # Применяем порог уверенности
            confidence_threshold = 0.3
            results_df = results_df[results_df['confidence'] >= confidence_threshold]

            predictions.append({
                'file': file.filename,
                'results': results_df.to_dict(orient='records')
            })

            # Отрисовка результатов на изображении
            filtered_image = np.array(img)
            for _, row in results_df.iterrows():
                label = row['name']
                conf = row['confidence']
                xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
                color = (0, 255, 0)
                cv2.rectangle(filtered_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 1)
                cv2.putText(filtered_image, f'{label} {conf:.2f}', (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Сохранение изображения с детекцией
            output_image_path = os.path.join(upload_folder, f'detected_{file.filename}')
            cv2.imwrite(output_image_path, filtered_image)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"predictions": predictions, "uploaded_files": uploaded_files})