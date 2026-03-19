from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

# Используем абсолютные импорты
from model.generate_data import generate_data
from model.create_model import create_model
from model.train_model import train_and_evaluate

def weights_to_c_array(model):
    """Конвертирует веса в C-массив"""
    c_code = "// Веса нейросети для STM32\n"
    c_code += "// Сгенерировано автоматически\n\n"
    
    weights = model.get_weights()
    layer_names = ['fc1', 'fc2', 'output']
    
    weight_idx = 0
    for i, layer_name in enumerate(layer_names):
        w = weights[weight_idx].flatten()
        c_code += f"const float {layer_name}_weights[] = {{\n    "
        for j, val in enumerate(w):
            c_code += f"{val:.6f}f"
            if j < len(w)-1:
                c_code += ", "
            if (j+1) % 8 == 0:
                c_code += "\n    "
        c_code += "\n};\n\n"
        
        b = weights[weight_idx+1].flatten()
        c_code += f"const float {layer_name}_bias[] = {{\n    "
        for j, val in enumerate(b):
            c_code += f"{val:.6f}f"
            if j < len(b)-1:
                c_code += ", "
        c_code += "\n};\n\n"
        
        weight_idx += 2
    
    return c_code

def save_model_files(model):
    """Сохраняет модель и веса"""
    model.save('stm32_signal_classifier.h5')
    
    c_code = weights_to_c_array(model)
    with open('model_weights.h', 'w') as f:
        f.write(c_code)
    
    print("\n✅ Модель обучена и сохранена!")
    print("📁 Файлы:")
    print("   - stm32_signal_classifier.h5 (полная модель)")
    print("   - model_weights.h (веса для STM32 в формате C)")

def plot_results(history, X_test):
    """Визуализирует результаты"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Точность модели')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    t = np.linspace(0, 2*np.pi, 20)
    plt.plot(t, X_test[0], label='Шум (класс 0)')
    plt.plot(t, X_test[1], label='Синус (класс 1)')
    plt.title('Примеры сигналов')
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Основной код выполняется только при прямом запускееее
    model, history, test_loss, test_acc = train_and_evaluate()
    print(f"\nТочность на тесте: {test_acc:.4f}")
    
    save_model_files(model)
    
    # Проверяем размер модели
    total_params = model.count_params()
    print(f"\n📊 Параметры модели:")
    print(f"   Всего параметров: {total_params}")
    print(f"   Память под веса: {total_params * 4 / 1024:.2f} KB (float32)")
    print(f"   Память под веса: {total_params * 2 / 1024:.2f} KB (float16)")
    
    # Визуализируем
    X_train, y_train = generate_data(2000)
    X_test, y_test = generate_data(400)
    plot_results(history, X_test)