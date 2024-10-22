import os
import queue
import json
import struct
import sys
import pyttsx3
import sounddevice as sd
import pvporcupine
import vosk
import requests
import numpy as np
import pyaudio
import random
import webbrowser
import time
import datetime
from gtts import gTTS
import subprocess
import librosa
from sklearn.ensemble import RandomForestClassifier
from playsound import playsound
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from pydub import AudioSegment
from pydub.playback import play as play_audio
import openai
import tkinter as tk
import pygame
from apscheduler.schedulers.background import BackgroundScheduler  # Scheduler for alarms
from bs4 import BeautifulSoup  # BeautifulSoup for web scraping
from words import *  # Assuming this is a custom module for triggers

# Constants
CDIR = os.getcwd()
SAMPLE_RATE = 16000
WAKE_WORD = 'alexa'
PORCUPINE_KEY = 'GnXXnq/UstTO0dR002/GPYP+nMsFPgHJzzTvecoJg99XI13YFxd4sg=='
OPENAI_API_KEY = '1d8f677e76454e4c995f48f8648395ff.3b0477bae71a2471'
WEATHER_API_URL = 'https://api.weatherapi.com/v1/current.json'
WEATHER_API_KEY = 'f7a51032f4c134dec171910751b38ff4'

# Initialize TTS and PyGame mixer
engine = pyttsx3.init()
pygame.mixer.init()

# Queue for audio data
q = queue.Queue()

# Инициализация глобальных переменных
samplerate = 16000
device = sd.default.device

# Load Vosk models
model_ru = vosk.Model('model-small-ru')
rec_ru = vosk.KaldiRecognizer(model_ru, SAMPLE_RATE)

model_en = vosk.Model('model-small-en')
rec_en = vosk.KaldiRecognizer(model_en, SAMPLE_RATE)

# Initialize Porcupine for wake word detection
porcupine = pvporcupine.create(
    access_key=PORCUPINE_KEY,
    keywords=[WAKE_WORD]
)

# Инициализация распознавателя Vosk
rec = vosk.KaldiRecognizer(model_ru, SAMPLE_RATE)


# Callback function for audio stream
def callback(indata, frames, time_info, status):
    """Функция обратного вызова для обработки аудиопотока."""
    if status:
        print(f"Status: {status}", flush=True)
    q.put(bytes(indata))

# Определение функций для времени и погоды
def speak_time():
    """Озвучить текущее время."""
    speak(f"Сейчас {datetime.datetime.now().strftime('%H:%M')}")


# Запуск будильника и задач по расписанию
def alarm(text):
    """Озвучить будильник."""
    speak(text)

def start_scheduler():
    scheduler = BackgroundScheduler()

    # Будильник в 6:00
    scheduler.add_job(lambda: alarm("Подъем! Время вставать!"), 'cron', hour=6, minute=0)

    # Озвучка времени каждый час
    scheduler.add_job(speak_time, 'cron', minute=0)  # Каждый час ровно

    scheduler.start()


def get_weather_data(location='Moscow'):
    """Получение данных о погоде через API."""
    url = f"{WEATHER_API_URL}?key={WEATHER_API_KEY}&q={location}&aqi=no"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        current_weather = data['current']
        condition = current_weather['condition']['text']
        temperature = current_weather['temp_c']
        return {"description": condition, "temperature": temperature}
    else:
        speak("Извините, я не смог получить информацию о погоде.")
        return {}

def speak_weather():
    """Озвучить текущую погоду."""
    weather_data = get_weather_data()
    if weather_data:
        speak(f"Сейчас на улице {weather_data['description']} с температурой {weather_data['temperature']} градусов.")

# Toggle sound
def toggle_sound(mute=True):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMute(1 if mute else 0, None)
    speak("Звук отключен." if mute else "Звук включен.")

# Распознавание пола по голосу
def extract_features(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    average_mfccs = np.mean(mfccs, axis=1)
    return average_mfccs

def load_gender_model():
    """Загрузка модели для распознавания пола."""
    audio_files = [os.path.join(CDIR, "audio1.mp3"), os.path.join(CDIR, "audio2.mp3"),
                   os.path.join(CDIR, "audio3.mp3"),
                   os.path.join(CDIR, "audio4.mp3")]
    x_train = np.array([extract_features(audio_file) for audio_file in audio_files])
    y_train = np.array(['male', 'female', 'male', 'female'])

    if x_train.size > 0 and y_train.size > 0:
        model = RandomForestClassifier(n_estimators=100)
        model.fit(x_train, y_train)
        return model
    else:
        print("Ошибка: неверные данные для обучения модели.")
        return None

gender_model = load_gender_model()

def detect_gender(audio):
    """Определение пола по голосу."""
    features = extract_features(audio)
    gender = gender_model.predict([features])[0]
    return gender

# Управление компьютером
def offpc():
    """Выключение компьютера."""
    speak("Выключаю компьютер.")
    os.system("shutdown /s /t 1")  # Эта команда для Windows

def restartpc():
    """Перезагрузка компьютера."""
    speak("Перезагружаю компьютер.")
    os.system("shutdown /r /t 1")  # Эта команда для Windows


# Команды для управления устройством и выполнения задач
def data_set():
    return {
        'привет': lambda: speak('и тебе, привет'),
        'какая сейчас погода': speak_weather,
        'какая погода на улице': speak_weather,
        'что там на улице': speak_weather,
        'сколько градусов': speak_weather,
        'сколько сейчас времени': speak_time,
        'который час': speak_time,
        'запусти браузер': lambda: webbrowser.open("https://www.google.com"),
        'открой браузер': lambda: webbrowser.open("https://www.google.com"),
        'играть': pygame,
        'хочу поиграть в игру': pygame,
        'запусти игру': pygame,
        'выключи компьютер': offpc,
        'отключись': offpc
    }


def triggers():
    return set(data_set().keys())


# Словарь команд и действий
commands = {
    'what is the weather': "weather",
    'open browser': "open_browser",
    'what time is it': "time",
    'ask': "ask_openai",
    'mute sound': "sound_off",
    'unmute sound': "sound_on",
    'привет': "greet",
    'какая сейчас погода': "weather",
    'какая погода на улице': "weather",
    'что там на улице': "weather",
    'сколько градусов': "weather",
    'запусти браузер': "open_browser",
    'открой браузер': "open_browser",
    'открой интернет': "open_browser",
    'играть': "play_game",
    'хочу поиграть в игру': "play_game",
    'запусти игру': "play_game",
    'посмотреть фильм': "open_browser",
    'выключи компьютер': "shutdown",
    'отключись': "shutdown",
    'как у тебя дела': "passive",
    'что делаешь': "passive",
    'работаешь': "passive",
    'расскажи анекдот': "joke",
    'ты тут': "passive",
    'how are you doing today': "passive",
    'good night': "passive",
    'пока': "passive"
}


# Определяем действия для каждой команды
actions = {
    'greet': lambda: speak('и тебе, привет'),
    'weather': speak_weather,
    'time': speak_time,
    'open_browser': lambda: webbrowser.open("https://www.google.com"),
    'play_game': lambda: speak('Запускаю игру...'),
    'shutdown': offpc
}

# Обработка команд и выполнение соответствующих действий
def recognize_and_execute(data, vectorizer, clf):
    """Распознает команду из аудио и выполняет соответствующую функцию."""
    trg = triggers().intersection(data.split())
    if trg:
        # Очистка команды от триггерного слова
        data = data.replace(list(trg)[0], '').strip()

    # Векторизация входного текста
    command_vector = vectorizer.transform([data])

    # Прогнозирование действия
    predicted_action = clf.predict(command_vector)[0]

    # Выполнение действия
    command_actions = data_set()  # Получаем словарь команд
    if predicted_action in command_actions:
        command_actions[predicted_action]()  # Выполняем команду

    else:
        process_command(data)  # Обрабатываем неизвестные команды

# Обработка нераспознанных команд через GPT
def process_command(command_text):
    """Обработка команд, которые не найдены в словаре."""
    if "time" in command_text:
        speak(f"The time is {datetime.datetime.now().strftime('%H:%M')}")
    elif "weather" in command_text:
        speak_weather()
    elif "browser" in command_text:
        webbrowser.open("https://www.google.com")
    else:
        # Если команда не распознана, отправляем запрос в GPT
        gpt_response = get_gpt_response(command_text)
        translated_response = translate(gpt_response)
        speak(translated_response)

# Распознавание речи и синтез
def get_gpt_response(prompt):
    """Получение ответа от GPT для неизвестных команд."""
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Ошибка при вызове GPT: {e}")
        return "Извините, я не смог получить ответ от GPT."

def translate(text):
    """Перевод текста на русский язык."""
    try:
        translated_text = GoogleTranslator(source='auto', target='ru').translate(text)
        return translated_text
    except Exception as e:
        print(f"Ошибка перевода: {e}")
        return text

def speak(text):
    """Озвучивание текста."""
    print(f"Alexa: {text}")
    engine.say(text)
    engine.runAndWait()


def va_speak(text):
    tts = gTTS(text=text, lang="ru")
    tts.save("response.mp3")
    AudioSegment.from_file("response.mp3").export("response.wav", format="wav")
    play(AudioSegment.from_wav("response.wav"))


# Основная функция
def main():
    """Основная программа для настройки распознавания голоса и обработки команд."""
    # Инициализация векторизатора и классификатора
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(list(commands.keys()))

    clf = LogisticRegression()
    clf.fit(vectors, list(commands.values()))

    # Запуск прослушивания аудиопотока
    with sd.RawInputStream(samplerate=samplerate, blocksize=4000, device=device,
                           dtype='int16', channels=1, callback=callback):
        print("Слушаю...")
        speak("Слушаю...")

        while True:
            if not q.empty():  # Проверяем наличие данных в очереди
                data = q.get()
                if rec.AcceptWaveform(data):  # Работа с распознаванием Vosk
                    result = rec.Result()
                    text = json.loads(result).get('text', '')

                    if text:
                        print(f"Распознано: {text}")
                        recognize_and_execute(text, vectorizer, clf)  # Распознаем и выполняем команду
            time.sleep(0.2)

if __name__ == "__main__":
    main()
