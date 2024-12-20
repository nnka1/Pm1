import numpy as np  # Импортируем библиотеку NumPy для работы с массивами
import matplotlib.pyplot as plt  # Импортируем библиотеку Matplotlib для построения графиков
from sklearn.model_selection import train_test_split  # Импортируем функцию для разделения данных на тренировочный и тестовый наборы
from sklearn.metrics import accuracy_score, classification_report  # Импортируем функции для оценки модели

# Данные - координаты точек для двух классов
x1 = np.array([0.38, 1.42, 0.55, 1.34, 1.76, 1.62, 0.83, 0.84, 1.77, 1.06])
y1 = np.array([1.79, 0.54, 0.34, 0.678, 1.64, 0.92, 1.49, 0.3, 0.7, 0.99])
x2 = np.array([3.9, 6.14, 6.1, 2.11, 3.23, 1.62, 1.88])
y2 = np.array([4.93, 4.95, 0.97, 0.77, 0.43, 4.61, 0.25])

# Преобразуем данные в матрицы
features1 = np.column_stack((x1, y1))  # Объединяем x1 и y1 в матрицу (столбцы)
features2 = np.column_stack((x2, y2))  # Объединяем x2 и y2 в матрицу (столбцы)

# Объединяем данные двух классов в одну матрицу features
features = np.concatenate((features1, features2))  # Объединяем features1 и features2 по вертикали

# Создаем вектор меток классов (0 для первого класса, 1 для второго)
labels = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))))

# Стандартизация признаков (центрирование и нормирование)
features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

# Разделение данных на тренировочный и тестовый наборы (80% тренировка, 20% тест)
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)  # random_state для воспроизводимости результатов

# Гиперпараметры модели
learning_rate = 0.01  # Скорость обучения
epochs = 1500  # Количество эпох обучения
lambda_reg = 0.1  # Коэффициент L2-регуляризации


# Сигмоидная функция (активационная функция для логистической регрессии)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Функция для предсказания класса
def predict(X, weights):
    z = np.dot(X, weights[:-1]) + weights[-1]  # Линейная комбинация признаков и весов
    return sigmoid(z)  # Применение сигмоиды для получения вероятности

# Функция потерь (с L2-регуляризацией)
def loss(y_true, y_pred, weights, lambda_reg):
    epsilon = 1e-15  # Для предотвращения логарифма от нуля
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Ограничиваем предсказания для устойчивости
    regularization_term = (lambda_reg / 2) * np.sum(weights[:-1]**2)  # L2-регуляризация
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) + regularization_term  # Функция бинарной кросс-энтропии


# Инициализация весов (случайные значения)
weights = np.random.randn(features.shape[1] + 1)  # +1 для смещения

# Обучение модели с помощью градиентного спуска
for epoch in range(epochs):
    y_pred = predict(features_train, weights)  # Предсказания на тренировочном наборе
    error = y_pred - labels_train  # Разница между предсказаниями и истинными значениями
    gradient = np.dot(features_train.T, error) / len(features_train)  # Вычисление градиента
    gradient = np.concatenate(([np.mean(error)], gradient))  # Добавляем градиент для смещения

    weights[:-1] -= learning_rate * (gradient[1:] + lambda_reg * weights[:-1])  # Обновление весов (с L2-регуляризацией)
    weights[-1] -= learning_rate * gradient[0]  # Обновление смещения

    if (epoch + 1) % 100 == 0:
        print(f"Эпоха {epoch+1}/{epochs}, Потери: {loss(labels_train, y_pred, weights, lambda_reg)}")

# Оценка модели на тестовом наборе
y_pred_test = predict(features_test, weights)  # Предсказания на тестовом наборе
y_pred_test_classes = (y_pred_test > 0.5).astype(int)  # Преобразование вероятностей в классы (0 или 1)
accuracy = accuracy_score(labels_test, y_pred_test_classes)  # Вычисление точности
print(f"\nТочность на тестовом наборе: {accuracy:.4f}")
print("Отчет по классификации:")
print(classification_report(labels_test, y_pred_test_classes, zero_division=0))  # Отчет о классификации (precision, recall, f1-score)

# Визуализация результатов (только для 2D данных)
if features.shape[1] == 2:
    plt.figure(figsize=(8, 6))
    plt.scatter(features[labels == 0, 0], features[labels == 0, 1], label='Класс 0', marker='o')  # Точки первого класса
    plt.scatter(features[labels == 1, 0], features[labels == 1, 1], label='Класс 1', marker='x')  # Точки второго класса

    xx = np.linspace(np.min(features[:, 0]), np.max(features[:, 0]), 100)  # Диапазон значений для построения линии
    yy = -(weights[0] * xx + weights[2]) / weights[1]  # Уравнение разделяющей линии
    plt.plot(xx, yy, 'r-', label="Разделяющая линия")  # Построение разделяющей линии

    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title('Логистическая регрессия')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Визуализация невозможна для более чем 2 признаков.")
