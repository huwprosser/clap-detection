import os
import argparse
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


def augment_folder(path):
    for filename in os.listdir(path):
        if filename.endswith(".wav") and "shifted" not in filename:
            audio_file = os.path.join(path, filename)
            for i in range(-1, 1):  # Create 2 pitch-shifted versions
                shifted, sr = pitch_shift(audio_file, i)  # Shift pitch by i semitones
                new_filename = f"{os.path.splitext(filename)[0]}_shifted{i}.wav"
                sf.write(os.path.join(path, new_filename), shifted, sr)

                # Create 3 volume-augmented versions of each pitch-shifted version
                for db_change in [-10, 0, 10]:  # Change volume by -10, 0, or 10 dB
                    volume_changed = change_volume(
                        os.path.join(path, new_filename), db_change
                    )
                    volume_changed_filename = (
                        f"{os.path.splitext(new_filename)[0]}_volume{db_change}.wav"
                    )
                    volume_changed.export(
                        os.path.join(path, volume_changed_filename), format="wav"
                    )


if __name__ == "__main__":
    
    augment_folder("data/background_noise")
