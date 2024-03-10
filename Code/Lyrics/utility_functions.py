import json
from lyricsgenius import Genius
import configparser
import re
import pandas as pd
import os


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
    return load_json('../../Data/west_coast.json')


def load_east_coast_json():
    return load_json('../../Data/east_coast.json')


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

    # Convert text to lowercase
    text = text.lower()

    # Replace specified characters with the desired character or an empty string
    text = text.replace('.', '').replace('-', ' ').replace("’", '')
    text = text.replace("?", '').replace("!", '').replace("*", 'i')
    text = text.replace('&#8217;', '').replace(',', '').replace('&amp;', '&')
    text = text.replace('\n', ' ')

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
        frequency_distribution = {word: row['Lyrics'].count(word) for word in set(row['Lyrics'])}
        word_scores = {word: (concreteness_ratings.get(word, 0), freq) for word, freq in frequency_distribution.items()}
        correctness_score = calculate_concreteness_score(word_scores)
        # Assign the new value to a new column for that row
        dataframe.at[index, 'Correctness'] = correctness_score
    return dataframe


def filter_dataframe_by_artist(df, artist):
    return df[df['Artist'] == artist]


def filter_dataframe_by_album(df, year):
    return df[df['Album Release Year'] == year]
