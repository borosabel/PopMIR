import json
from lyricsgenius import Genius
import configparser
import re
import pandas as pd
import os
import librosa
import numpy as np
from scipy.stats import gmean, kurtosis
import madmom

FRAME_LENGTH = 2048
HOP_LENGTH = 512

def get_set_of_artists(data):
    artists = []
    for key, value in data.items():
        artists.append(value['artist'])
    return list(set(artists))


def filter_data_between_years(data, start_year, end_year):
    filtered_data = {}
    for key, value in data.items():
        year = int(value['date'])
        if start_year <= year < end_year:
            filtered_data[key] = value
    return filtered_data


def load_east_west_json():
    east_coast = load_east_coast_json()
    west_coast = load_west_coast_json()
    return east_coast, west_coast


def load_west_coast_json():
    return load_json('../../Data/rolling_stone_100_west_coast.json')


def load_east_coast_json():
    return load_json('../../Data/rolling_stone_100_east_coast.json')


def save_json(save_name, dictionary):
    with open(f'{save_name}', 'w') as json_file:
        json.dump(dictionary, json_file, ensure_ascii=False, indent=4)
    print(f'Data has been saved to "{save_name}"')


def load_json(file_name):
    with open(f'{file_name}', 'r') as json_file:
        loaded_data = json.load(json_file)
    return loaded_data


def cleanup(text):
    """
    Input: string
    Function to clean up the text.
    Removes text within square brackets and parentheses, converts to lowercase,
    and removes specified characters.
    Returns a cleaned string.
    """
    # Remove text within square brackets and parentheses
    text = re.sub(r"\[.*?\]", "", text)  # Removes [bracketed text]
    text = re.sub(r"\(.*?\)", "", text)  # Removes (parenthesized text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)

    # Replace multiple instances of punctuation with a single instance
    text = re.sub(r'([.!?,:;])\1+', r'\1', text)

    # Ensure a single space follows punctuation
    text = re.sub(r'([.!?,:;])([^\s])', r'\1 \2', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Replace curly quotes with straight quotes
    text = text.replace('“', '"').replace('”', '"')

    # Remove or replace nested quotes (depending on preference)
    # This pattern looks for a quote inside a quoted string and removes it
    text = re.sub(r'\"(.*?)\"(.*?)\"(.*?)\"', r'"\1\2\3"', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Convert text to lowercase
    text = text.lower()

    # Remove leading and trailing quotes
    text = text.strip('"')

    # Replace specified characters with the desired character or an empty string
    text = text.replace('.', '').replace('-', ' ').replace("’", '')
    text = text.replace("?", '').replace("!", '').replace("*", 'i')
    text = text.replace('&#8217;', '').replace(',', '').replace('&amp;', '&')
    text = text.replace('\n', '')

    # Here address the word related issues.
    text = text.replace('niggas', 'nigga').replace('niggaz', 'nigga')
    text = (text.replace('yo', 'yeah')
            .replace('yah ', 'yeah')
            .replace('ya', 'yeah')
            .replace('yea', 'yeah')
            .replace('yep', 'yeah')
            .replace('yeahhu', 'yeah')
            .replace('yeahhur', 'yeah')
            .replace('yeahh', 'yeah')
            .replace('yeahhr', 'yeah')
            .replace('yeahhh', 'yeah'))

    return text


def artist_cleanup(text):
    _text = text
    if 'feat.' in _text:
        _text = _text.split('feat.')[0]
    _text = cleanup(_text)
    return _text


def get_genius_object():
    config = configparser.ConfigParser()
    # Load the config file
    config.read('./config.ini')  # Update with the correct path to your config file
    # Get the API key
    client_access_token = config['API']['api_key']
    return Genius(client_access_token)


# --------------------------------DATAFRAME RELATED FUNCTIONS----------------------------------------
def load_txt_into_dataframe(path):
    """
    Input: path - string
    Function to recursively go trought on the folder structure
    and load all the .txt files into a dataframe.
    Used to load lyrics.
    """

    base_directory = path
    year_pattern = r'\(\d{4}\)'
    # Initialize a list to store data
    data = []

    # Loop through each root, directory, and file in the base directory
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            # Check if the file is a text file
            if file.endswith(".txt"):
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Open and read the contents of the text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    match = re.search(year_pattern, file_path)
                    if match:
                        extracted_year = match.group()
                # Append the file path, file name, and content to the data list
                data.append({"Artist": file_path.split('/')[-3], "Album": file_path.split('/')[-2],
                             "Album Release Year": int(extracted_year[1:-1]),
                             "Song": file.replace('.txt', ''), "Lyrics": content})

    return pd.DataFrame(data)


def load_audio_into_dataframe(path):
    """
    Input: path - string
    Function to recursively go through the folder structure
    and load metadata from all the audio files into a DataFrame.
    Used to load audio file metadata.
    """

    base_directory = path
    data = []

    # Loop through each root, directory, and file in the base directory
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            # Check if the file is an audio file (e.g., .wav, .mp3)
            if file.lower().endswith((".wav", ".mp3")):
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Load the audio file with librosa
                y, sr = librosa.load(file_path, sr=None)  # sr=None to preserve the native sampling rate

                # Get duration in seconds
                duration = librosa.get_duration(y=y, sr=sr)

                # Append the file path, file name, duration, and sample rate to the data list
                data.append({
                    "Artist": file_path.split('/')[-3],
                    "Album": file_path.split('/')[-2],
                    "Coast": file_path.split('/')[-4].lower(),
                    "Path": file_path,
                    "Song": file,
                    "Duration (s)": duration,
                    "Sample Rate (Hz)": sr
                })

    # Convert the list of dictionaries into a pandas DataFrame
    return pd.DataFrame(data)


def calculate_concreteness_score(word_scores):
    nominator = 0
    denomiantor = 0
    for key, value in word_scores.items():
        nominator += value[0] * value[1]
        denomiantor += value[1]

    return nominator / denomiantor


def calculate_correctness_score_of_tokens(dataframe, concreteness_ratings):
    for index, row in dataframe.iterrows():
        # Calculate new value for the current row
        frequency_distribution = {word: row['Tokens'].count(word) for word in set(row['Tokens'])}
        word_scores = {word: (concreteness_ratings.get(word, 0), freq) for word, freq in frequency_distribution.items()}
        correctness_score = calculate_concreteness_score(word_scores)
        # Assign the new value to a new column for that row
        dataframe.at[index, 'Correctness'] = correctness_score
    return dataframe


def word_count_of_text(text):
    return len(text.split(' '))


def unique_word_count_of_text(text):
    return len(list(set(text.split(' '))))


def filter_dataframe_by_artist(df, artist):
    return df[df['Artist'] == artist]


def filter_dataframe_by_album(df, year):
    return df[df['Album Release Year'] == year]


# --------------------------------DATAFRAME RELATED FUNCTIONS----------------------------------------

# --------------------------------LOADER FUNCTIONS----------------------------------------
def get_all_artists(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data


def get_all_lyrics_of_an_artist(artist_name, json_path):
    artists_data = get_all_artists(json_path)  # Assuming this function returns a dict-like object
    artist_data = artists_data[artist_name]

    lyrics_paths = [item['lyrics_path'] for item in artist_data if 'lyrics_path' in item]

    # Collect all DataFrames in a list first
    dfs = [load_txt_into_dataframe(path) for path in lyrics_paths]

    # Concatenate all DataFrames at once
    all_lyrics_df = pd.concat(dfs, ignore_index=True)

    return all_lyrics_df


def get_all_lyrics_of_an_artist_between_years(artist_name, json_path, start_year, end_year):
    artists_data = get_all_artists(json_path)  # Assuming this function returns a dict-like object
    artist_data = artists_data[artist_name]

    lyrics_paths = []

    for item in artist_data:
        if 'lyrics_path' in item:
            release_year = int(item['release_date'])
            if start_year <= release_year <= end_year:
                lyrics_paths.append(item['lyrics_path'])

    # Collect all DataFrames in a list first
    dfs = [load_txt_into_dataframe(path) for path in lyrics_paths]

    # Concatenate all DataFrames at once
    all_lyrics_df = pd.concat(dfs, ignore_index=True)

    return all_lyrics_df


def get_all_audio_of_an_artist(artist_name, json_path):
    artists_data = get_all_artists(json_path)  # Assuming this function returns a dict-like object
    artist_data = artists_data[artist_name]

    audio_paths = [item['audio_path'] for item in artist_data if 'audio_path' in item]

    # Collect all DataFrames in a list first
    dfs = [load_audio_into_dataframe(path) for path in audio_paths]

    # Concatenate all DataFrames at once
    all_audio_df = pd.concat(dfs, ignore_index=True)

    return all_audio_df


def get_all_audio_of_an_artist_between_years(artist_name, json_path, start_year, end_year):
    artists_data = get_all_artists(json_path)  # Assuming this function returns a dict-like object
    artist_data = artists_data[artist_name]

    audio_paths = []

    for item in artist_data:
        if 'audio_path' in item:
            release_year = int(item['release_date'])
            if start_year <= release_year <= end_year:
                audio_paths.append(item['audio_path'])

    # Collect all DataFrames in a list first
    dfs = [load_audio_into_dataframe(path) for path in audio_paths]

    # Concatenate all DataFrames at once
    all_audio_df = pd.concat(dfs, ignore_index=True)

    return all_audio_df


def get_all_artist_lyrics_between_years(json_path, start_year, end_year):
    artists_data = get_all_artists(json_path)

    lyrics_paths = []

    for item in artists_data:
        if 'lyrics_path' in item:
            release_year = int(item['release_date'])
            if start_year <= release_year <= end_year:
                lyrics_paths.append(item['lyrics_path'])

    dfs = [load_txt_into_dataframe(path) for path in lyrics_paths]
    all_lyrics_df = pd.concat(dfs, ignore_index=True)

    return all_lyrics_df


def get_all_artist_lyrics(json_path):
    artists_data = get_all_artists(json_path)

    lyrics_paths = []

    for item in artists_data.values():
        for i in item:
            if 'lyrics_path' in i:
                lyrics_paths.append(i['lyrics_path'])

    dfs = [load_txt_into_dataframe(path) for path in lyrics_paths]
    all_lyrics_df = pd.concat(dfs, ignore_index=True)

    return all_lyrics_df


def get_all_artist_audio_between_years(json_path, start_year, end_year):
    artists_data = get_all_artists(json_path)

    audio_paths = []

    for item in artists_data:
        if 'audio_path' in item:
            release_year = int(item['release_date'])
            if start_year <= release_year <= end_year:
                audio_paths.append(item['audio_paths'])

    dfs = [load_audio_into_dataframe(path) for path in audio_paths]
    all_audio_df = pd.concat(dfs, ignore_index=True)

    return all_audio_df


def get_all_artist_audio(json_path):
    artists_data = get_all_artists(json_path)

    audio_paths = []
    release_years_expanded = []
    dfs = []
    for item in artists_data.values():
        for i in item:
            if 'audio_path' in i:
                # Load each audio file into a DataFrame
                audio_df = load_audio_into_dataframe(i['audio_path'])
                dfs.append(audio_df)

                release_years_expanded.extend([i['release_date']] * len(audio_df))

    all_audio_df = pd.concat(dfs, ignore_index=True)
    all_audio_df['Release Year'] = release_years_expanded

    return all_audio_df


# --------------------------------LOADER FUNCTIONS---------------------------------------

# --------------------------------FEATURES FROM ISABELLA'S PAPER---------------------------------------

# Calculate spectral centroid to approximate brightness
def calculate_brightness(y, sr):
    S = np.abs(librosa.stft(y))
    power_spectrum = np.abs(S) ** 2
    frequencies = librosa.fft_frequencies(sr=sr)
    high_freq_mask = frequencies > 1500
    total_energy = np.sum(power_spectrum)
    high_freq_energy = np.sum(power_spectrum[high_freq_mask, :], axis=0)
    avg_high_freq_energy = np.mean(high_freq_energy)
    brightness_percentage = (avg_high_freq_energy / total_energy) * 100 if total_energy > 0 else 0

    return brightness_percentage


# Calculate energy in a specific frequency band (2000-4000 Hz)
def band_energy(y, sr):
    S, phase = librosa.magphase(librosa.stft(y))
    frequencies = librosa.fft_frequencies(sr=sr)
    band_mask = (frequencies >= 2000) & (frequencies <= 4000)
    band_energy = np.sqrt(np.mean(np.square(S[band_mask])))
    return band_energy


# Calculate envelope flatness
def envelope_flatness(y):
    envelope = librosa.feature.rms(y=y)[0]
    return gmean(envelope) / np.mean(envelope)


def envelope_kurtosis(y):
    # Compute the temporal envelope using the onset strength
    envelope = librosa.feature.rms(y=y)[0]
    env_kurtosis = kurtosis(envelope, fisher=True)
    return env_kurtosis


def calculate_envelope_quantile_range(y):
    envelope = librosa.feature.rms(y=y)[0]
    q_0_9 = np.quantile(envelope, 0.9)
    q_0_2 = np.quantile(envelope, 0.2)
    quantile_range = q_0_9 - q_0_2

    return quantile_range

def calculate_first_attack_time(y, sr=44100):
    # Get onset times in samples
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    if len(onset_times) == 0:
        return 0  # No onsets were detected
    # Assuming the first onset is the start of the first attack
    first_onset = onset_times[0]
    # Get the amplitude envelope
    envelope = librosa.feature.rms(y=y)
    # Find the peak after the first onset (end of attack phase)
    first_onset_sample = librosa.time_to_samples([first_onset], sr=sr)[0]
    peak_index = np.argmax(envelope[0][first_onset_sample:]) + first_onset_sample
    peak_time = librosa.samples_to_time([peak_index], sr=sr)[0]
    # Calculate the duration of the first attack phase
    first_attack_duration = peak_time - first_onset
    return first_attack_duration

def calculate_harmonic_energy(y):
    # Separate the harmonic part from the noise
    y_harmonic = librosa.effects.harmonic(y)
    # Calculate the RMS energy of the harmonic component
    rms_energy = np.sqrt(np.mean(y_harmonic**2))
    return rms_energy


def calculate_harmonic_percussive_ratio(y):
    # Separate the harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Calculate the RMS energy of the harmonic component
    rms_harmonic = np.sqrt(np.mean(y_harmonic**2))

    # Calculate the RMS energy of the percussive component
    rms_percussive = np.sqrt(np.mean(y_percussive**2))

    # Calculate the ratio of RMS energies
    if rms_percussive != 0:
        ratio = rms_harmonic / rms_percussive
    else:
        ratio = float('inf')  # Avoid division by zero

    return ratio


def calculate_high_frequency_ratio(y, sr):
    # Perform the Short-Time Fourier Transform (STFT)
    S = np.abs(librosa.stft(y))
    frequencies = librosa.fft_frequencies(sr=sr)

    high_freq_indices = np.where(frequencies > 1000)[0]
    band_freq_indices = np.where((frequencies >= 250) & (frequencies <= 400))[0]

    S_high = S[high_freq_indices, :]
    S_band = S[band_freq_indices, :]

    rms_high = np.sqrt(np.mean(np.square(S_high)))

    rms_band = np.sqrt(np.mean(np.square(S_band)))

    if rms_band != 0:
        ratio = rms_high / rms_band
    else:
        ratio = float('inf')  # Avoid division by zero

    return ratio


import pyloudnorm as pyln
import soundfile as sf
def calculate_loudness_sone(file_path):
    # Load audio file
    data, rate = sf.read(file_path)

    # Create a Meter instance with ITU-R BS.1770-4 algorithm
    meter = pyln.Meter(rate)  # specify sample rate

    # Measure loudness
    loudness = meter.integrated_loudness(data)

    # Convert from LUFS to sones approximately (very rough approximation)
    sones = 2 ** ((loudness - 40) / 10)  # This is a simplified approximation

    return sones


def calculate_low_energy(y):
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms)
    low_energy_frames = np.sum(rms < avg_rms)
    low_energy_percentage = (low_energy_frames / len(rms)) * 100  # percentage
    return low_energy_percentage


def find_max_rms_position(y, sr):
    rms_values = librosa.feature.rms(y=y)[0]
    max_rms_index = np.argmax(rms_values)
    max_rms_position = librosa.frames_to_time([max_rms_index], sr=sr, hop_length=512)[0]
    return max_rms_position


def find_max_rms_value(y):
    rms_values = librosa.feature.rms(y=y)[0]
    max_rms_value = np.max(rms_values)
    return max_rms_value


def count_segments_based_on_rms(y, threshold=0.01):
    rms = librosa.feature.rms(y=y)[0]
    rms_diff = np.abs(np.diff(rms))
    change_points = np.where(rms_diff > threshold)[0]
    num_segments = len(change_points) + 1

    return num_segments


def calculate_percussive_energy(y):
    _, y_percussive = librosa.effects.hpss(y)
    rms_energy = np.sqrt(np.mean(np.square(y_percussive)))
    return rms_energy


def calculate_rms_energy(y):
    rms_energy = librosa.feature.rms(y=y)[0]
    return rms_energy


def calculate_average_rms_energy(y):
    rms_energy = librosa.feature.rms(y=y)[0]
    average_rms_energy = np.mean(rms_energy)
    return average_rms_energy

def calculate_average_spectral_centroid(y, sr):
    centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    average_centroid = np.mean(centroids)
    return average_centroid


from scipy.stats import entropy
def calculate_spectral_entropy(y):
    S = np.abs(librosa.stft(y))
    psd = np.mean(S**2, axis=1)
    psd_norm = psd / np.sum(psd)
    spectral_entropy = entropy(psd_norm)
    return spectral_entropy

def average_calculate_spectral_flatness(y):
    flatness = librosa.feature.spectral_flatness(y=y)
    return np.mean(flatness)

def zero_crossing_rate(y):
    zcrs = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    zcr_mean = np.mean(zcrs)
    zcr_std = np.std(zcrs)
    return zcr_mean, zcr_std



# --------------------------------FEATURES FROM ISABELLA'S PAPER---------------------------------------
