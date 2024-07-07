import random
import torchaudio
import torch as th
from tqdm import tqdm
import pandas as pd

SAMPLING_RATE = 44100
HOP_LENGTH = 512  # Define the hop length (e.g., 512 samples)
ONSETS_ABS_ERROR_RATE_IN_SECONDS = 0.050
WIN_LENGTHS = [1024, 2048, 4096]
WIN_SIZES = [0.023, 0.046, 0.093]
N_MELS = 80
F_MIN = 27.5
F_MAX = 16000
NUM_FRAMES = 15
FRAME_LENGTH = HOP_LENGTH * (NUM_FRAMES - 1)

def preprocess_audio_train(df, max_non_gunshot_samples=1):
    spectrograms = []
    sample_rates = []
    labels = []

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        file_path = row['filename']
        num_gunshots = row['num_gunshots']
        gunshot_times = row['gunshot_location_in_seconds']

        waveform, sample_rate = torchaudio.load(file_path)

        if num_gunshots == 0:
            # Select random segments for non-gunshot data
            for _ in range(min(max_non_gunshot_samples, int(waveform.size(1) / FRAME_LENGTH))):
                segment = select_random_segment(waveform, sample_rate, FRAME_LENGTH)
                mel_specgram = calculate_melbands(segment[0], sample_rate)
                spectrograms.append(mel_specgram)
                sample_rates.append(sample_rate)
                labels.append(0)
        else:
            # Process gunshot data
            for gunshot_time in gunshot_times:
                segment = select_gunshot_segment(waveform, sample_rate, gunshot_time, FRAME_LENGTH)
                mel_specgram = calculate_melbands(segment[0], sample_rate)
                spectrograms.append(mel_specgram)
                sample_rates.append(sample_rate)
                labels.append(1)

    return spectrograms, sample_rates, labels

def select_random_segment(waveform, sample_rate, frame_length):
    total_duration = waveform.size(1) / sample_rate
    segment_length = frame_length

    if total_duration * sample_rate <= frame_length:
        return waveform

    start_time = random.uniform(0, total_duration - (frame_length / sample_rate))
    start_sample = int(start_time * sample_rate)
    end_sample = start_sample + segment_length

    return waveform[:, start_sample:end_sample]

def select_gunshot_segment(waveform, sample_rate, gunshot_time, frame_length):
    start_time = max(0, gunshot_time - (frame_length / sample_rate) / 2)
    start_sample = int(start_time * sample_rate)
    end_sample = start_sample + int(frame_length)
    return waveform[:, start_sample:end_sample]

def calculate_melbands(waveform, sample_rate):
    mel_specs = []
    for wl in WIN_LENGTHS:
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=wl,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX
        )(waveform)
        mel_specs.append(mel_spectrogram)
    return th.log10(th.stack(mel_specs) + 1e-08)


def preprocess_audio(files):
    spectrograms = []
    sample_rates = []

    for file_path in tqdm(files):
        waveform, sample_rate = torchaudio.load(file_path)
        mel_specgram = calculate_melbands(waveform[0], sample_rate)
        spectrograms.append(mel_specgram)
        sample_rates.append(sample_rate)

    return spectrograms, sample_rates

def make_frames(X, y):
    X_frames, y_frames = [], []

    X_frames.append(X)
    y_frames.append(y)

    return X_frames, y_frames
