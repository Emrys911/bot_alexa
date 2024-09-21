import os
import sys
from concurrent.futures import ThreadPoolExecutor

import openai
import pvporcupine
import pyttsx3
import sounddevice
from telegram import (Application, CommandHandler, MessageHandler)
from telegram import Update

from skills import start_scheduler, listen_for_command, respond_to_command, offline_recognition, online_recognition

# Initialize the TTS engine (Pyttsx3)
engine = pyttsx3.init()
engine.setProperty('rate', 180)

# Load API keys and tokens from environment variables for security
OPENAI_API_KEY = os.getenv("1d8f677e76454e4c995f48f8648395ff.3b0477bae71a2471")
TELEGRAM_BOT_TOKEN = os.getenv("6701996903:AAG86hAkmORtKPQ-HYpZLquM9hiBNGoEKkg")

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize wake-word engine (Picovoice Porcupine) with predefined wake word
try:
    porcupine = pvporcupine.create(access_key="YOUR_ACCESS_KEY", keywords=["mila"], sensitivities=[1])
except Exception as e:
    print(f"Error initializing Porcupine: {e}")
    sys.exit(1)

# Setup PyAudio for wake-word detection
pa = sounddevice.Sounddevice()
audio_stream = pa.open(rate=porcupine.sample_rate, channels=1, format=sounddevice.paInt16, input=True,
                       frames_per_buffer=porcupine.frame_length)


# ... (Other functions like speaker, alarm, and weather)

def recognize_command(audio, recognizer, Triggers):
    """Handle recognition using both online and offline methods."""
    command = online_recognition(recognizer, audio)

    if not command:  # If online recognition fails, use offline
        audio_data = audio.get_raw_data()
        command = offline_recognition(audio_data)

    if command and any(trigger in command for trigger in Triggers):
        command = command.replace('Mila', '').strip()
        respond_to_command(command)


# ==================== Telegram Bot Integration =======================

async def start(update: Update) -> None:
    """Start command for the Telegram bot."""
    await update.message.reply_text('Привет! Я Mila, как могу помочь?')


async def handle_message(update: Update) -> None:
    """Handle any text messages sent to the Telegram bot."""
    user_message = update.message.text.lower()
    await update.message.reply_text(f'Вы сказали: {user_message}')
    respond_to_command(user_message)


def telegram_bot():
    """Start the Telegram bot."""
    application = Application.builder().token("6701996903:AAG86hAkmORtKPQ-HYpZLquM9hiBNGoEKkg").build()
    application.add_handler(CommandHandler())
    application.add_handler(MessageHandler())

    application.run_polling()


if __name__ == "__main__":
    triggers = ["Alexa", "Bot Alexa"]
    start_scheduler()

    # Use ThreadPoolExecutor to run both the Telegram bot and the voice assistant concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(telegram_bot)
        executor.submit(listen_for_command, triggers)
