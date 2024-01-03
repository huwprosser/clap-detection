import sounddevice as sd
import numpy as np
import os
import time
from predict import transform_audio, load_model
import torch
from scipy.io.wavfile import write

# Set the duration of each chunk
chunk_duration = .3  # in seconds
num_samples = 100
sample_rate = 44100  # in Hz
dtype = np.int16

# Calculate the number of samples in each chunk
chunk_samples = int(chunk_duration * sample_rate)

# Create a stream
stream = sd.InputStream(callback=None, channels=1, samplerate=sample_rate, dtype=dtype)

# Define the directory where the chunks will be saved
directory = "./"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

model = load_model("audio_classifier.pth")

# Start the stream
with stream:
    print("Recording...")
    for i in range(num_samples):  
        # Read a chunk from the stream
        chunk, overflowed = stream.read(chunk_samples)

        # Save the chunk to a WAV file in the specified directory
        filename = os.path.join(directory, f"temp.wav")
        write(filename, sample_rate, chunk)

        # Wait for the next chunk
        time.sleep(chunk_duration)

        spec = transform_audio(filename)

        output = model(spec)
        prediction = torch.argmax(output, dim=1)
        probabilities = torch.softmax(output, dim=1)
        predicted_prob = probabilities[0][prediction].item()

        classes = {0: "Noise", 1: "Clap"}

        print(
            f"Predicted class: {classes[prediction.item()]}, Probability: {predicted_prob}"
        )


print("Recording finished.")
