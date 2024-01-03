import os
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import Resize
import torchaudio.transforms as T

def get_wav_files(directory):
    return [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith(".wav")
    ]

class AudioDataset(Dataset):
    def __init__(self, noise_dir, clap_dir):
        noise_files = get_wav_files(noise_dir)
        clap_files = get_wav_files(clap_dir)

        self.noise_dir = noise_dir
        self.clap_dir = clap_dir
        self.file_list = noise_files + clap_files
        self.labels = [0] * len(os.listdir(noise_dir)) + [1] * len(os.listdir(clap_dir))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx, n_mels=128, n_fft=400, hop_length=200):
        waveform, sample_rate = torchaudio.load(self.file_list[idx])
        spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )(waveform)
        spec = Resize((256, 256))(spec)
        spec = (spec - spec.mean()) / spec.std()  # normalize the spectrogram
        label = self.labels[idx]
        return spec, torch.tensor(label)

