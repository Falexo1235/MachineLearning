# Используем официальный образ Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файл requirements.txt и .whl файл в контейнер
COPY dist/*.whl .

# Устанавливаем зависимости и наш .whl файл
RUN pip install --no-cache-dir *.whl

# Копируем остальные файлы проекта
COPY . .

# Команда для запуска приложения
ENTRYPOINT ["python", "-m", "machinelearning.model"]