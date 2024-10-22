import os
import sys
from concurrent.futures import ThreadPoolExecutor

import openai
import pvporcupine
import pyttsx3
import sounddevice
import telegram
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
    porcupine = pvporcupine.create(access_key="ACCESS_KEY", keywords=["alexa"], sensitivities=[1])
except Exception as e:
    print(f"Error initializing Porcupine: {e}")
    sys.exit(1)

# Setup sounddevice for wake-word detection
audio_stream = sd.InputStream(samplerate=porcupine.sample_rate, channels=1, dtype='int16', callback=None)
audio_stream.start()

# Global variable to track wake-word activation and conversation context
wake_word_active = False
conversation_context = "default"


# Function to handle wake-word detection
def detect_wake_word():
    global wake_word_active

    pcm_data = audio_stream.read(porcupine.frame_length)[0]
    pcm_data = pcm_data.flatten()

    if porcupine.process(pcm_data) >= 0:
        wake_word_active = True
        engine.say("I'm listening")
        engine.runAndWait()
        print("Wake word 'alexa' detected.")


# Function to recognize and execute commands
def recognize_command(audio, recognizer):
    global conversation_context, wake_word_active

    if wake_word_active:  # Process the command only if wake word has been detected
        command = online_recognition(recognizer, audio)

        if not command:  # If online recognition fails, use offline
            audio_data = audio.get_raw_data()
            command = offline_recognition(audio_data)

        if command and any(trigger in command for trigger in trigger):
            command = command.replace('alexa', '').strip()
            print(f"Recognized command: {command}")
            # Update the conversation context
            update_conversation_context(command)
            # Respond to the recognized command
            respond_to_command(command)
            wake_word_active = False  # Reset wake word activation after processing


# Function to update conversation context based on command
def update_conversation_context(command):
    global conversation_context

    if "weather" in command:
        conversation_context = "weather"
    elif "news" in command:
        conversation_context = "news"
    elif "timer" in command:
        conversation_context = "timer"
    else:
        conversation_context = "general conversation"

    print(f"Updated conversation context to: {conversation_context}")


# ==================== Telegram Bot Integration =======================

async def start(update: Update) -> None:
    """Start command for the Telegram bot."""
    await update.message.reply_text('Привет! Я Alexa, как могу помочь?')


async def handle_message(update: Update) -> None:
    """Handle any text messages sent to the Telegram bot."""
    user_message = update.message.text.lower()
    await update.message.reply_text(f'Вы сказали: {user_message}')
    # You can handle the command or message here
    respond_to_command(command)


def telegram_bot():
    """Start the Telegram bot."""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add command handler
    application.add_handler(CommandHandler('start', start))
    # Add message handler for handling regular messages
    application.add_handler(MessageHandler(None, handle_message))

    application.run_polling()


# Main function to run the voice assistant and Telegram bot concurrently
if __name__ == "__main__":
    keywords = triggers = ["alexa"]
    start_scheduler()

    # Use ThreadPoolExecutor to run both the Telegram bot and the voice assistant concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(telegram_bot)
        executor.submit(listen_for_command, triggers, detect_wake_word)
