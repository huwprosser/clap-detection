import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torchvision.transforms import Resize
from model import AudioClassifier


def load_model(model_path):
    model = AudioClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def transform_audio(audio_path, n_mels=128, n_fft=400, hop_length=200):
    waveform, sample_rate = torchaudio.load(audio_path)
    spec = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )(waveform)
    spec = Resize((256, 256))(spec)
    spec = (spec - spec.mean()) / spec.std()  # normalize the spectrogram
    return spec.unsqueeze(0)  # add a dimension for batch size


def predict(model, audio_path):
    spec = transform_audio(audio_path)
    output = model(spec)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()


if __name__ == "__main__":
    model = load_model("audio_classifier.pth")

    while True:
        audio_path = input("Enter the path to an audio file: ")
        prediction = predict(model, audio_path)
        print("Noise = 0, Clap = 1")
        print(f"The predicted class for {audio_path} is {prediction}")
