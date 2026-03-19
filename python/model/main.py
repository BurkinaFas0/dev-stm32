import numpy as np
import tensorflow as tf
import tensorflow as tf
import matplotlib.pyplot as plt

from generate_data.generate_data import generate_data



# Генерируем данные данные
X_train, y_train = generate_data(2000)   # данные для обучения модели
X_test, y_test = generate_data(400)      # Используется только для проверки
                                            # Модель НИКОГДА не видит эти данные во время обучения

# Создаем очень маленькую модель для STM32
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(20,)),  # 20 отсчетов сигнала
    tf.keras.layers.Dense(16, activation='relu'),  # Первый слой
    tf.keras.layers.Dense(8, activation='relu'),   # Второй слой
    tf.keras.layers.Dense(1, activation='sigmoid') # Выход для бинарной классификации
])

# Компилируем модель
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Смотрим размер модели
model.summary()

# Обучаем. Внутри происходит: Прямой проход --> Считаем ошибку --> Обратное распространение --> Обновляем веса (Adam)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Оцениваем
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nТочность на тесте: {test_acc:.4f}")

# Сохраняем веса и архитектуру
model.save('stm32_signal_classifier.h5')
# Когда вы сохраняете модель после обучения, внутри одного файла оказываются
# Архитектура модели — описание слоев (20 → 16 → 8 → 1), их типов и функций активации.
# Веса — те самые настроенные 481 число, которые модель "выучила".
# Конфигурация оптимизатора — состояние оптимизатора (например, Adam) на момент сохранения, 
# чтобы можно было продолжить обучение позже.
# Функция потерь и метрики — информация о том, как модель была скомпилирован

# Конвертируем веса в массив C для STM32
def weights_to_c_array(model):
    c_code = "// Веса нейросети для STM32\n"
    c_code += "// Сгенерировано автоматически\n\n"
    
    # Получаем веса
    weights = model.get_weights()
    
    layer_names = ['fc1', 'fc2', 'output']
    
    weight_idx = 0
    for i, layer_name in enumerate(layer_names):
        # Веса
        w = weights[weight_idx].flatten()
        c_code += f"const float {layer_name}_weights[] = {{\n    "
        for j, val in enumerate(w):
            c_code += f"{val:.6f}f"
            if j < len(w)-1:
                c_code += ", "
            if (j+1) % 8 == 0:
                c_code += "\n    "
        c_code += "\n};\n\n"
        
        # Смещения (bias)
        b = weights[weight_idx+1].flatten()
        c_code += f"const float {layer_name}_bias[] = {{\n    "
        for j, val in enumerate(b):
            c_code += f"{val:.6f}f"
            if j < len(b)-1:
                c_code += ", "
        c_code += "\n};\n\n"
        
        weight_idx += 2
    
    return c_code

# Генерируем C-код
c_code = weights_to_c_array(model)

# Сохраняем в файл
with open('model_weights.h', 'w') as f:
    f.write(c_code)

print("\n✅ Модель обучена и сохранена!")
print("📁 Файлы:")
print("   - stm32_signal_classifier.h5 (полная модель)")
print("   - model_weights.h (веса для STM32 в формате C)")

# Визуализируем результаты
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Точность модели')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
# Покажем примеры сигналов
t = np.linspace(0, 2*np.pi, 20)
plt.plot(t, X_test[0], label='Шум (класс 0)')
plt.plot(t, X_test[1], label='Синус (класс 1)')
plt.title('Примеры сигналов')
plt.xlabel('Время')
plt.ylabel('Амплитуда')
plt.legend()

plt.tight_layout()
plt.show()

# Проверим размер модели для STM32
total_params = model.count_params()
print(f"\n📊 Параметры модели:")
print(f"   Всего параметров: {total_params}")
print(f"   Память под веса: {total_params * 4 / 1024:.2f} KB (float32)")
print(f"   Память под веса: {total_params * 2 / 1024:.2f} KB (float16)")