# Используем Python 3.10 как базовый образ
FROM python:3.10

# Обновляем систему и устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        libasound2-dev \
        portaudio19-dev \
        build-essential \
        ffmpeg \
        libopenblas-dev \
        libfftw3-dev \
        espeak \
        espeak-ng \
        python3-dev \
        python3-gi \
    || cat /var/log/apt/term.log && \
    rm -rf /var/lib/apt/lists/*

# Установите рабочий каталог
WORKDIR /app

# Копируем файл зависимостей и приложение
COPY requirements.txt ./
COPY . /app

# Обновляем pip и устанавливаем Python зависимости
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip check

# Set up PulseAudio and ALSA configuration for the container
RUN useradd -ms /bin/bash appuser
USER appuser

# Копируем модели в контейнер
COPY model-small-en /app/model-small-en
COPY model-small-ru /app/model-small-ru

# Устанавливаем переменные окружения для корректной работы звука
ENV AUDIODEV=hw:0,0

# Запуск приложения
CMD ["python", "main_app.py"]
