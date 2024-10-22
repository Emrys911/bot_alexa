#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from main_app import va_speak

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'bot_alexa.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)



def data_set():
    """Returns a dictionary mapping voice commands to their respective actions."""
    return {
        'привет': lambda: va_speak('и тебе, привет'),
        'какая сейчас погода': lambda: va_speak('weather сейчас скажу'),
        'какая погода на улице': lambda: va_speak('weather боишься замерзнуть?'),
        'что там на улице': lambda: va_speak('weather сейчас гляну'),
        'сколько градусов': lambda: va_speak('weather можешь выглянуть в окно, но сейчас проверю'),
        'запусти браузер': lambda: va_speak('browser запускаю браузер'),
        'открой браузер': lambda: va_speak('browser открываю браузер'),
        'открой интернет': lambda: va_speak('browser интернет активирован'),
        'играть': lambda: va_speak('game лишь бы баловаться'),
        'хочу поиграть в игру': lambda: va_speak('game а нам лишь бы баловаться'),
        'запусти игру': lambda: va_speak('game запускаю, а нам лишь бы баловаться'),
        'посмотреть фильм': lambda: va_speak('browser сейчас открою браузер'),
        'выключи компьютер': lambda: va_speak('offpc отключаю'),
        'отключись': lambda: va_speak('offbot отключаюсь'),
        'как у тебя дела': lambda: va_speak('passive работаю в фоне, не переживай'),
        'что делаешь': lambda: va_speak('passive жду очередной команды, мог бы и сам на кнопку нажать'),
        'работаешь': lambda: va_speak('passive как видишь'),
        'расскажи анекдот': lambda: va_speak('passive вчера помыл окна, теперь у меня рассвет на 2 часа раньше'),
        'ты тут': lambda: va_speak('passive вроде, да'),
        'how are you doing today': lambda: va_speak('passive nice, and what about you'),
        'good night': lambda: va_speak('passive bye, bye'),
        'пока': lambda: va_speak('passive Пока')
    }


def triggers():
    """Returns a set of voice command triggers."""
    return {'привет', 'какая сейчас погода', 'какая погода на улице', 'что там на улице', 'сколько градусов',
            'запусти браузер', 'открой браузер', 'открой интернет', 'играть', 'хочу поиграть в игру',
            'запусти игру', 'посмотреть фильм', 'выключи компьютер', 'отключись', 'как у тебя дела',
            'что делаешь', 'работаешь', 'расскажи анекдот', 'ты тут', 'how are you doing today',
            'good night', 'пока'}


# Пример вызова функций
if __name__ == '__main__':
    # Пример использования
    data = data_set()
