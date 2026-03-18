#!/bin/bash

echo "Создание виртуального окружения..."
python3 -m venv python/venv

echo "Активация виртуального окружения..."
source python/venv/bin/activate

echo "Установка зависимостей..."
pip install --upgrade pip
pip install -r python/requirements


echo "Готово!"
echo "Теперь выполните: source python/venv/bin/activate"