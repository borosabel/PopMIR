import json
from lyricsgenius import Genius
import configparser

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
    Returns a string.
    """
    _text = text.lower()
    _text = _text.replace('.', '')
    _text = _text.replace('.', '')
    _text = _text.replace('-', ' ')
    _text = _text.replace("’", '')
    _text = _text.replace("?", '')
    _text = _text.replace("!", '')
    _text = _text.replace("*", 'i')
    _text = _text.replace('&#8217;', '')
    _text = _text.replace(',', '')
    _text = _text.replace('&amp;', '&')
    return _text

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