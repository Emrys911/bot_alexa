import os
import json
import queue
import random

import openai
import pvporcupine
import pyttsx3
import requests
import sounddevice as sd
import vosk
from deep_translator import GoogleTranslator
from playsound import playsound
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import words

# Constants
SAMPLE_RATE = 16000  # Adjust based on your microphone's capabilities
BLOCK_SIZE = 512  # Block size for audio capture
WEATHER_API_KEY = 'f7a51032f4c134dec171910751b38ff4'
WEATHER_API_URL = 'https://api.weatherapi.com/v1/current.json'

# Environment Variables
access_key = os.getenv('ACCESS_KEY')

# Porcupine for wake word detection (alexa)
keyword_path = r"D:\repositories\bot_alexa\bot_alexa\keywords\alexa.ppn"
porcupine = pvporcupine.create(
    access_key=access_key,
    keywords=['alexa'],
    sensitivities=[1]
)

# Initialize pyttsx3 engine for speech synthesis
engine = pyttsx3.init()

# Load Vosk speech recognition models for English and Russian
model_path_en = r"/app/model-small-en"
model_path_ru = r"/app/model-small-ru"
model_en = vosk.Model(model_path_en)
model_ru = vosk.Model(model_path_ru)
rec_en = vosk.KaldiRecognizer(model_en, SAMPLE_RATE)
rec_ru = vosk.KaldiRecognizer(model_ru, SAMPLE_RATE)

# Queue for handling audio data
q = queue.Queue()


# Function to process the captured audio data
def process_audio(data, language='en'):
    rec = rec_en if language == 'en' else rec_ru
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        return result.get("text", "")
    else:
        result = json.loads(rec.PartialResult())
        return result.get("partial", "")


CDIR = r"D:\repositories\bot_alexa\bot_alexa"


def play(phrase):
    filename = f"{CDIR}\\sounds\\"

    match phrase:
        case "greet":
            filename += f"greet1.wav"  # Так как у вас только один файл для приветствия "greet1"

        case "ok":
            filename += f"ok{random.choice([1, 2, 3, 4])}.wav"  # Случайный выбор между ok1, ok2, ok3, ok4

        case "shutdown":
            filename += "shutdown1.wav"  # Для выключения только один файл "shutdown1"

        case "remark":
            filename += "remark1.wav"  # Для замечаний только один файл "remark1"

        case "reboot":
            filename += "reboot1.wav"  # Для перезагрузки только один файл "reboot1"

    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()


# Callback function to capture microphone input and put it into the queue
def audio_callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    q.put(indata.copy())  # Store captured audio in the queue


# Function to listen to microphone input and stream to speakers or process it
def va_listen(callback, language='en'):
    input_device = 2  # AMD Audio Device
    output_device = 4  # Realtek(R) Audio

    # Capture audio from microphone using RawInputStream
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, device=input_device, dtype='int16', channels=1,
                           callback=audio_callback) as mic_stream:
        # Stream audio from input to output (speakers) for real-time playback or processing
        with sd.Stream(device=(input_device, output_device), samplerate=SAMPLE_RATE, channels=1) as stream:
            print("Listening and processing audio...")

            while True:
                # Capture audio data from the microphone stream
                data, overflowed = mic_stream.read(BLOCK_SIZE)  # Specify BLOCK_SIZE (e.g., 8000 frames)
                if overflowed:
                    print("Audio buffer overflow")

                # Continuously process audio from the queue
                while not q.empty():
                    data = q.get()  # Get audio data from the queue
                    text = process_audio(data, language)  # Process audio data for speech recognition
                    if text:
                        callback(text)  # Use the callback to handle recognized text (e.g., perform actions)

            # Continuously process audio from the queue
            while True:
                data = q.get()  # Get audio data from the queue
                text = process_audio(data, language)  # Process audio data for speech recognition
                if text:
                    callback(text)  # Use the callback to handle recognized text (e.g., perform actions)


# Example callback function
def my_callback(text):
    print(f"Recognized text: {text}")


# Function to get weather info using the weather API
def get_weather():
    response = requests.get(WEATHER_API_URL, params={'key': WEATHER_API_KEY, 'q': 'auto:ip'})
    data = response.json()
    temp_c = data['current']['temp_c']
    condition = data['current']['condition']['text']
    return f"The current temperature is {temp_c}°C with {condition}."


# Function for translation using GoogleTranslator
def translator(text, target_lang="en"):
    try:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text


# Function to get response from GPT model
def gpt_answer(message):
    model_engine = "text-davinci-003"
    max_tokens = 128
    prompt = translator(message, target_lang="en")
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.9
    )
    return translator(completion.choices[0].text, target_lang="ru")


# Text-to-speech function
def speak(message):
    engine.say(message)
    engine.runAndWait()


# Function to filter and normalize recognized text
def filter_cmd(raw_voice):
    return raw_voice.lower().strip()


# Function to recognize gender from audio data
def recognize_gender(audio_data):
    models, utils = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_vad', source='github')
    get_speech_timestamps, _, read_audio, _, _ = utils
    wav = torch.tensor(audio_data, dtype=torch.float32)
    speech_timestamps = get_speech_timestamps(wav, models)
    gender = 'male' if len(speech_timestamps) % 2 == 0 else 'female'
    return gender


# Function to recognize commands and execute corresponding actions
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


# Function to respond to voice commands
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


# Main function to start the voice assistant
def main():
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(list(words.data_set.keys()))
    clf = LogisticRegression()
    clf.fit(vectors, list(words.data_set.values()))
    print("Listening for wake word...")

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
