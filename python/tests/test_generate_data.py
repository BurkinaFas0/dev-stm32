import pytest
import numpy as np
import sys
import os

# Добавляем путь к корневой директории проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.generate_data import generate_data

@pytest.mark.unit
def test_generate_data_returns_correct_shape():
    """Проверяет, что функция возвращает данные правильной формы"""
    n_samples = 100
    X, y = generate_data(n_samples)
    
    assert X.shape == (n_samples, 20)
    assert y.shape == (n_samples,)

@pytest.mark.unit
def test_generate_data_returns_binary_labels():
    """Проверяет, что метки только 0 или 1"""
    X, y = generate_data(100)
    
    assert np.all((y == 0) | (y == 1))

@pytest.mark.unit
def test_generate_data_returns_balanced_classes():
    """Проверяет, что классы примерно сбалансированы"""
    X, y = generate_data(1000)
    
    class_0_count = np.sum(y == 0)
    class_1_count = np.sum(y == 1)
    
    # Проверяем, что разница не более 20%
    assert abs(class_0_count - class_1_count) < 200

@pytest.mark.unit
def test_generate_data_signal_class_has_sine_wave():
    """Проверяет, что сигналы класса 1 похожи на синусоиду"""
    X, y = generate_data(100)
    
    # Берем первый сигнал класса 1
    signal_1 = X[y == 1][0]
    
    # Проверяем, что сигнал примерно в диапазоне [-1.5, 1.5]
    assert np.all(np.abs(signal_1) < 1.5)
    
    # Проверяем, что есть периодичность (приблизительно)
    assert abs(signal_1[0] - signal_1[10]) < 0.5

@pytest.mark.unit
def test_generate_data_reproducible_with_seed():
    """Проверяет, что можно воспроизвести результаты с seed"""
    np.random.seed(42)
    X1, y1 = generate_data(10)
    
    np.random.seed(42)
    X2, y2 = generate_data(10)
    
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)