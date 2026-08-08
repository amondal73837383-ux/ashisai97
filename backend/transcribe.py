"""
Speech-to-text using OpenAI's Whisper API.
Takes raw audio bytes (e.g. from a mic recorder widget) and returns text.
"""

from io import BytesIO
from openai import OpenAI


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav", client: OpenAI = None) -> str:
    """
    audio_bytes: raw audio file bytes (wav/mp3/m4a etc.)
    filename: needs a valid extension so the API can infer the format.
    """
    client = client or OpenAI()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = filename  # OpenAI SDK reads this to detect format

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    return transcript.text
