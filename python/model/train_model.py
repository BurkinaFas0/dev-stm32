import numpy as np
import tensorflow as tf
from .generate_data import generate_data
from .create_model import create_model

# В train_model.py
def train_and_evaluate(X_train=None, y_train=None, X_test=None, y_test=None, 
                       epochs=50, validation_data=None):
    """Обучает модель и возвращает историю и результаты"""
    if X_train is None or y_train is None:
        X_train, y_train = generate_data(2000)
        X_test, y_test = generate_data(400)
    
    model = create_model()
    
    if validation_data:
        # Используем явно переданные validation данные
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=validation_data,
            verbose=0
        )
    else:
        # Используем validation_split
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    return model, history, test_loss, test_acc
