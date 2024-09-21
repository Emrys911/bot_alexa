import os
import json
import queue
import openai
import pvporcupine
import pyttsx3
import requests
import sounddevice as sd
import vosk
from deep_translator.detection import config
from playsound import playsound
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import torch
from deep_translator import GoogleTranslator
import words

# Constants
SAMPLE_RATE = 16000
DEVICE = 0
WEATHER_API_KEY = 'f7a51032f4c134dec171910751b38ff4'
WEATHER_API_URL = 'https://api.weatherapi.com/v1/current.json'

# Получение ключа из переменной окружения
access_key = os.getenv('ACCESS_KEY')

# Путь к скачанному файлу ключевого слова
keyword_path = r"D:\Proekti\bot_alexa\keywords\hey_alexa.ppn"

# Создание объекта Porcupine с использованием переменной окружения для ключа доступа
porcupine = pvporcupine.create(
    access_key=access_key,
    keywords=['Hey Alexa'],
    sensitivities=[1]
)

# Инициализация движка pyttsx3 для синтеза речи
engine = pyttsx3.init()

model_path_en = r"D:\Proekti\bot_alexa\model-small-en"
model_path_ru = r"D:\Proekti\bot_alexa\model-small-ru"

# Initialize models
model_en = vosk.Model(model_path_en)
model_ru = vosk.Model(model_path_ru)

# Create recognizers for each model
rec_en = vosk.KaldiRecognizer(model_en, SAMPLE_RATE)
rec_ru = vosk.KaldiRecognizer(model_ru, SAMPLE_RATE)

# Queue for audio data
q = queue.Queue()


def process_audio(data, language='en'):
    rec = rec_en if language == 'en' else rec_ru
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        return result.get("text", "")
    else:
        result = json.loads(rec.PartialResult())
        return result.get("partial", "")


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata)


def va_listen(callback, language='en'):
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, device=DEVICE, dtype='int16', channels=1,
                           callback=audio_callback):
        while True:
            data = q.get()
            text = process_audio(data, language)
            if text:
                callback(text)


# Example callback function
def my_callback(text):
    print(f"Recognized text: {text}")


def get_weather():
    response = requests.get("https://api.openweathermap.org/data/2.5/weather", params={
        'key': "f7a51032f4c134dec171910751b38ff4",
        'q': 'auto:ip'
    })
    data = response.json()
    temp_c = data['current']['temp_c']
    condition = data['current']['condition']['text']
    return f"The current temperature is {temp_c}°C with {condition}."


def translator(text, target_lang="en"):
    try:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def gpt_answer(message):
    model_engine = "text-davinci-003"
    max_tokens = 128
    prompt = translator(message, target_lang="en")
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.9,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return translator(completion.choices[0].text, target_lang="ru")


def play(filename):
    playsound(filename)


def speak(message):
    engine.say(message)
    engine.runAndWait()


def filter_cmd(raw_voice):
    return raw_voice.lower().strip()


def recognize_gender(audio_data):
    models, utils = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_vad', source='github')
    get_speech_timestamps, _, read_audio, _, _ = utils
    wav = torch.tensor(audio_data, dtype=torch.float32)
    speech_timestamps = get_speech_timestamps(wav, models)
    gender = 'male' if len(speech_timestamps) % 2 == 0 else 'female'
    return gender


def recognize_and_execute(data, vectorizer, clf):
    trg = words.triggers.intersection(data.split())
    if not trg:
        return
    data = data.replace(list(trg)[0], '')
    text_vec = vectorizer.transform([data]).toarray()[0]
    answer = clf.predict([text_vec])[0]
    func_name = answer.split()[0]
    speak(answer.replace(func_name, ''))
    exec(func_name + '()')


def va_respond(voice, recognize_cmd=None, vectorizer=None, clf=None):
    print(f"Recognized: {voice}")
    cmd = recognize_cmd(filter_cmd(voice))

    if isinstance(cmd, dict) and 'cmd' in cmd:

        if cmd['cmd'] in config.VA_CMD_LIST.keys():
            recognize_and_execute(cmd['cmd'], vectorizer, clf)
        else:
            gpt_result = gpt_answer(voice)
            speak(gpt_result)
    else:
        print("Error: 'cmd' is not in the correct format.")


def main():
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(list(words.data_set.keys()))

    clf = LogisticRegression()
    clf.fit(vectors, list(words.data_set.values()))

    print("Listening...")

    while True:
        pcm = b''  # Initialize an empty byte string
        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            gender = recognize_gender(pcm)
            greeting = "Yes, sir" if gender == "male" else "Yes, madam"
            speak(greeting)
            va_listen(lambda voice: va_respond(voice, vectorizer=vectorizer, clf=clf))


if __name__ == '__main__':
    main()
