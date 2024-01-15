import os
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

# Set the duration of each chunk
chunk_duration = 1  # in seconds
num_samples = 300
sample_rate = 44100  # in Hz
dtype = np.int16
interval = 1  # in seconds

# Calculate the number of samples in each chunk
chunk_samples = int(chunk_duration * sample_rate)

# Create a stream
stream = sd.InputStream(callback=None, channels=1, samplerate=sample_rate, dtype=dtype)

# Define the directory where the chunks will be saved
directory = "data/noise2"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Start the stream
with stream:
    print("Recording...")
    for i in range(num_samples):  # Record 10 chunks for testing
        print("Ready...", flush=True)
        time.sleep(0.2)
        print(f"Chunk {i+1} of {num_samples}.", flush=True)
        # Read a chunk from the stream
        chunk, overflowed = stream.read(chunk_samples)

        # Save the chunk to a WAV file in the specified directory
        filename = os.path.join(directory, f"bg_{i}.wav")
        write(filename, sample_rate, chunk)

        # Wait for the next chunk
        time.sleep(chunk_duration)


print("Recording finished.")
