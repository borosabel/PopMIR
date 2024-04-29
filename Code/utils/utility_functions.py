import json
from lyricsgenius import Genius
import configparser
import re
import pandas as pd
import os
import librosa


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
                    "FilePath": file_path,
                    "FileName": file,
                    "Duration (s)": duration,
                    "Sample Rate (Hz)": sr,
                    # You can add more features here
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

def get_all_artist_audio_between_years(json_path, start_year, end_year):
    artists_data = get_all_artists(json_path)

    audio_paths = []

    for item in artists_data:
        if 'audio_path' in item:
            release_year = int(item['release_date'])
            if start_year <= release_year <= end_year:
                audio_paths.append(item['audio_paths'])

    dfs = [load_audio_into_dataframe(path) for path in audio_paths]
    all_lyrics_df = pd.concat(dfs, ignore_index=True)

    return all_lyrics_df
# --------------------------------LOADER FUNCTIONS----------------------------------------