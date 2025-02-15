import random
import numpy as np
import pandas as pd
import torchaudio
import torch as th
from tqdm import tqdm
import librosa
import os
from pydub import AudioSegment
from pydub.playback import play

SAMPLING_RATE = 44100
HOP_LENGTH = 512  # Define the hop length (e.g., 512 samples)
ONSETS_ABS_ERROR_RATE_IN_SECONDS = 0.050
WIN_LENGTHS = [1024, 2048, 4096]
WIN_SIZES = [0.023, 0.046, 0.093]
N_MELS = 80
F_MIN = 27.5
F_MAX = 16000
NUM_FRAMES = 30
FRAME_LENGTH = HOP_LENGTH * (NUM_FRAMES - 1)

def get_next_file_index(output_folder, prefix):
    existing_files = [f for f in os.listdir(output_folder) if f.startswith(prefix)]
    if not existing_files:
        return 1
    indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
    return max(indices) + 1

def generate_data_samples(music_df, gunshot_df, number_of_samples_w_gunshots, number_of_samples_wo_gunshots, gunshot_volume_increase_dB=5):
    records = []
    output_folder = './gunshot_dataset'

    os.makedirs(output_folder, exist_ok=True)

    next_gunshot_index = get_next_file_index(output_folder, 'with_gunshot_')
    next_no_gunshot_index = get_next_file_index(output_folder, 'without_gunshot_')

    # Generate samples with gunshots
    for i in tqdm(range(number_of_samples_w_gunshots), desc="Generating Samples with Gunshots"):

        # Randomly select a music and a gunshot
        selected_music_row = music_df.sample(n=1).iloc[0]
        selected_gunshot_row = gunshot_df.sample(n=1).iloc[0]

        # Load the selected audio files
        music_audio = AudioSegment.from_file(selected_music_row['filename'])
        gunshot_audio = AudioSegment.from_file(selected_gunshot_row['filename'])

        # Increase the volume of the gunshot audio
        min_gunshot_volume_increase_dB = -5
        max_gunshot_volume_increase_dB = 10
        random_volume_increase_dB = random.uniform(min_gunshot_volume_increase_dB, max_gunshot_volume_increase_dB)
        gunshot_audio = gunshot_audio + random_volume_increase_dB

        # Gunshot start timestamps in the original audio in seconds
        gunshot_start_timestamps_sec = selected_gunshot_row['gunshot_location_in_seconds']
        gunshot_start_timestamps_ms = [t * 1000 for t in gunshot_start_timestamps_sec]  # Convert to milliseconds

        # Calculate gunshot duration and extra music duration
        gunshot_audio = gunshot_audio[gunshot_start_timestamps_ms[0]:]  # Cut the gunshot audio to start from the first occurrence
        gunshot_duration_ms = len(gunshot_audio)
        extra_music_duration = 5000  # Additional 5 seconds of music
        music_segment_duration = gunshot_duration_ms + 2000 + extra_music_duration

        # Ensure the segment fits within the music duration
        if len(music_audio) > music_segment_duration:
            music_start_time = random.randint(0, len(music_audio) - music_segment_duration)
        else:
            music_start_time = 0
            music_segment_duration = len(music_audio)

        # Cut out the segment from the original music
        music_segment = music_audio[music_start_time:music_start_time + music_segment_duration]

        # Overlay gunshot audio at the specified timestamps
        combined_audio = music_segment
        gunshot_positions_in_segment = [2000 + (ts - gunshot_start_timestamps_ms[0]) for ts in gunshot_start_timestamps_ms]  # Start first at 2 seconds

        for pos in gunshot_positions_in_segment:
            if pos < music_segment_duration:
                combined_audio = combined_audio.overlay(gunshot_audio, position=pos)

        # Create a descriptive output filename with song and gunshot base names
        music_base = os.path.splitext(os.path.basename(selected_music_row['filename']))[0]
        gunshot_base = os.path.splitext(os.path.basename(selected_gunshot_row['filename']))[0]
        output_filename = f'{output_folder}/with_gunshot_{music_base}_{gunshot_base}_{next_gunshot_index}.mp3'

        combined_audio.export(output_filename, format='mp3')
        next_gunshot_index += 1

        # Save the information in the records list
        records.append({
            'filename': output_filename,
            'source_song': selected_music_row['filename'],  # Add source song info
            'source_gunshot': selected_gunshot_row['filename'],  # Add source gunshot info
            'gunshot_location_in_seconds': [2 + (ts - gunshot_start_timestamps_ms[0]) / 1000 for ts in gunshot_start_timestamps_ms],  # Convert to seconds
            'num_gunshots': len(gunshot_start_timestamps_ms),
            'label': 1,
            'type': 'with_gunshot'
        })

    # Generate samples without gunshots
    for i in tqdm(range(number_of_samples_wo_gunshots), desc="Generating Samples without Gunshots"):
        # Randomly select a music
        selected_music_row = music_df.sample(n=1).iloc[0]

        music_audio = AudioSegment.from_file(selected_music_row['filename'])

        # Determine a random segment duration (let's use a fixed duration of 10 seconds for consistency)
        segment_duration = 10000  # 10 seconds in milliseconds

        # Ensure the segment fits within the music duration
        if len(music_audio) > segment_duration:
            start_time = random.randint(0, len(music_audio) - segment_duration)
        else:
            start_time = 0
            segment_duration = len(music_audio)

        # Cut out the segment from the original music
        music_segment = music_audio[start_time:start_time + segment_duration]

        # Create a descriptive output filename for the without gunshot case
        music_base = os.path.splitext(os.path.basename(selected_music_row['filename']))[0]
        output_filename = f'{output_folder}/without_gunshot_{music_base}_{next_no_gunshot_index}.mp3'

        music_segment.export(output_filename, format='mp3')
        next_no_gunshot_index += 1

        # Save the information in the records list
        records.append({
            'filename': output_filename,
            'source_song': selected_music_row['filename'],  # Add source song info
            'source_gunshot': None,  # No gunshot in this sample
            'gunshot_location_in_seconds': [],
            'num_gunshots': 0,
            'label': 0,
            'type': 'without_gunshot'
        })

    # Create a DataFrame from the records
    records_df = pd.DataFrame(records)
    return records_df


def overlay_gunshot_and_music(music_row, gunshot_row):
    music = AudioSegment.from_file(music_row['filename'])
    gunshot = AudioSegment.from_file(gunshot_row['filename'])


def preprocess_audio_train(df, max_non_gunshot_samples=1):
    spectrograms = []
    sample_rates = []
    labels = []

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        file_path = row['filename']
        num_gunshots = row['num_gunshots']
        gunshot_times = row['gunshot_location_in_seconds']

        try:
            waveform, sample_rate = torchaudio.load(file_path)

            if num_gunshots == 0:
                for _ in range(min(max_non_gunshot_samples, int(waveform.size(1) / FRAME_LENGTH))):
                    segment = select_random_segment(waveform, sample_rate, FRAME_LENGTH)
                    mel_specgram = calculate_melbands(segment[0], sample_rate)
                    spectrograms.append(mel_specgram)
                    sample_rates.append(sample_rate)
                    labels.append(0)
            else:
                for gunshot_time in gunshot_times:
                    segment = select_gunshot_segment(waveform, sample_rate, gunshot_time, FRAME_LENGTH)
                    mel_specgram = calculate_melbands(segment[0], sample_rate)
                    spectrograms.append(mel_specgram)
                    sample_rates.append(sample_rate)
                    labels.append(1)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

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

def select_gunshot_segment(waveform, sample_rate, gunshot_time, frame_length, max_shift_sec=0):
    """
    Selects a segment of audio around the gunshot, allowing for a random shift in the gunshot position.

    Parameters:
        waveform (Tensor): The audio waveform tensor.
        sample_rate (int): The sample rate of the audio.
        gunshot_time (float): The time of the gunshot in seconds.
        frame_length (int): The desired length of the segment (in samples).
        max_shift_sec (float): Maximum time in seconds by which to shift the gunshot position.

    Returns:
        Tensor: The selected segment of audio.
    """
    # Compute the amount of shift in time (in seconds) randomly within [-max_shift_sec, +max_shift_sec]
    random_shift = random.uniform(-max_shift_sec, max_shift_sec)

    # Adjust the gunshot start time with the random shift
    shifted_gunshot_time = gunshot_time + random_shift

    # Ensure the start time doesn't go out of bounds (start time should be >= 0)
    start_time = max(0, shifted_gunshot_time - (frame_length / sample_rate) / 2)

    # Ensure the end time doesn't exceed the waveform length
    start_sample = int(start_time * sample_rate)
    end_sample = start_sample + int(frame_length)

    # Ensure we don't exceed the total length of the waveform
    end_sample = min(end_sample, waveform.size(1))
    start_sample = max(0, end_sample - int(frame_length))  # Adjust start if needed

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

    mel_specs = th.log10(th.stack(mel_specs) + 1e-08)  # Shape: [3, 80, 15]

    # Calculate additional timbre features
    waveform_np = waveform.numpy()
    spectral_centroid = librosa.feature.spectral_centroid(y=waveform_np, sr=sample_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform_np, sr=sample_rate)
    spectral_flatness = librosa.feature.spectral_flatness(y=waveform_np)
    # MFCCs can be added similarly if required

    # Convert to tensors and ensure they match the shape along the time axis
    spectral_centroid = th.tensor(spectral_centroid, dtype=th.float32).unsqueeze(0)
    spectral_bandwidth = th.tensor(spectral_bandwidth, dtype=th.float32).unsqueeze(0)
    spectral_flatness = th.tensor(spectral_flatness, dtype=th.float32).unsqueeze(0)

    # Concatenate additional features along the feature axis
    additional_features = th.cat([spectral_centroid, spectral_bandwidth, spectral_flatness], dim=0)  # Shape: [3, 1, 15]

    # Resize additional features to match the mel spectrogram's feature dimension (if necessary)
    additional_features = additional_features.permute(1, 0, 2)

    # Ensure both tensors have the same shape along the feature and time dimensions
    mel_specs = mel_specs.permute(1, 0, 2)  # Shape: [80, 3, 15]
    combined_features = th.cat([mel_specs, additional_features], dim=0)
    combined_features = combined_features.permute(1, 0, 2)

    return combined_features


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