#!/usr/bin/env python3
import asyncio
import os
import tempfile
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from shazamio import Shazam

# RECORDING CONFIG
SAMPLE_RATE = 44100  # Hz
CHANNELS    = 1      # mono
DURATION    = 10     # seconds per snippet to analyze

async def recognize_snippet(wav_path: str):
    shazam = Shazam()
    out = await shazam.recognize_song(wav_path)
    if out.get('matches'):
        track = out.get('track', {})
        title = track.get('title', 'Unknown Title')
        subtitle = track.get('subtitle', 'Unknown Artist')
        print(f"✅ Recognized: '{title}' by {subtitle}")
        return True
    else:
        print("❌ No match, listening again...")
        return False

def record_audio(duration: int, fs: int, channels: int) -> np.ndarray:
    print(f"Recording for {duration} seconds…")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16')
    sd.wait()
    return recording

async def main_loop():
    while True:
        # 1. Record a snippet
        audio = record_audio(DURATION, SAMPLE_RATE, CHANNELS)

        # 2. Create a temp WAV file, then close it so we can write to it
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        # 3. Write the audio data
        write(tmp_path, SAMPLE_RATE, audio)

        # 4. Try to recognize
        success = await recognize_snippet(tmp_path)

        # 5. Clean up the temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        if success:
            break  # stop listening after first success

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting.")
