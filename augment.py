import os
import librosa
import soundfile as sf
from pydub import AudioSegment


def pitch_shift(audio_file, semitones):
    y, sr = librosa.load(audio_file)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    return y_shifted, sr


def change_volume(audio_file, db_change):
    audio = AudioSegment.from_wav(audio_file)
    return audio + db_change


import numpy as np


def add_noise(audio_file, noise_factor):
    y, sr = librosa.load(audio_file)
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * noise
    # Cast back to same data type
    augmented = augmented.astype(type(y[0]))
    return augmented, sr


def augment_folder(path):
    for filename in os.listdir(path):
        if filename.endswith(".wav") and "shifted" not in filename:
            audio_file = os.path.join(path, filename)

            for semitones in [-2, 2, 5]:
                y_shifted, sr = pitch_shift(audio_file, semitones)
                shifted_filename = (
                    f"{os.path.splitext(filename)[0]}_shifted{semitones}.wav"
                )
                sf.write(os.path.join(path, shifted_filename), y_shifted, sr)

    for filename in os.listdir(path):
        if filename.endswith(".wav") and "volume" not in filename:
            audio_file = os.path.join(path, filename)
            for db_change in [5, 10, 15]:
                volume_changed = change_volume(audio_file, db_change)
                volume_changed_filename = (
                    f"{os.path.splitext(filename)[0]}_volume{db_change}.wav"
                )
                volume_changed.export(
                    os.path.join(path, volume_changed_filename), format="wav"
                )


if __name__ == "__main__":
    augment_folder("data/noise2")
