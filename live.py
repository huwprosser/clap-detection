import sounddevice as sd
import numpy as np
import os
import time
from predict import transform_audio, load_model
import torch
from scipy.io.wavfile import write
from collections import deque

# Set the duration of each chunk and buffer
chunk_duration = 0.5  # in seconds
buffer_duration = 1.0  # in seconds
num_samples = 100
sample_rate = 44100  # in Hz
dtype = np.int16

# Calculate the number of samples in each chunk and buffer
chunk_samples = int(chunk_duration * sample_rate)
buffer_samples = int(buffer_duration * sample_rate)

# List all active devices
devices = sd.query_devices()
for i, device in enumerate(devices):
    print(f"{i}: {device['name']}")

# Ask for user input
device_index = int(input("Enter the index of the device you want to use: "))

# Create a stream with the selected device
stream = sd.InputStream(
    device=device_index, callback=None, channels=1, samplerate=sample_rate, dtype=dtype
)

# Define the directory where the chunks will be saved
directory = "./"

model = load_model("audio_classifier.pth")


# Create a rolling buffer
buffer = deque(maxlen=buffer_samples)  #

# Start the stream
with stream:
    print("Recording...")
    for i in range(num_samples):
        # Read a chunk from the stream
        chunk, overflowed = stream.read(chunk_samples)

        # Add the chunk to the buffer
        buffer.extend(chunk)

        # Save the buffer to a WAV file in the specified directory
        filename = os.path.join(directory, f"temp.wav")
        write(filename, sample_rate, np.array(buffer))

        spec = transform_audio(filename)

        output = model(spec)
        prediction = torch.argmax(output, dim=1)
        probabilities = torch.softmax(output, dim=1)
        predicted_prob = probabilities[0][prediction].item()

        classes = {0: "Noise", 1: "Clap"}

        if predicted_prob > 0.99 and prediction.item() == 1:
            print(f"👏 {predicted_prob * 100:.2f}%")
        else:
            print(".")

        # Wait for the next chunk
        time.sleep(chunk_duration)

# cup sounds
# slapping sounds
# me talking
# keyboard clicking
# mouse clicking


print("Recording finished.")
