import time
import random
import os
import re

from string import ascii_lowercase
from bs4 import BeautifulSoup
from urllib.request import urlopen


def save_file(
    path,
    text,
    replace=False
):
    if not replace:
        if os.path.exists(os.path.relpath(path + ".txt")):
            file = open(path + "_2" + ".txt", "w")
            file.write(text)
            file.close()
        else:
            file = open(path + ".txt", "w")
            file.write(text)
            file.close()
    else:
        file = open(path + ".txt", "w")
        file.write(text)
        file.close()

def sanitize_filename(filename):
    # Replace any character not a letter, number, underscore, or space with an underscore
    return re.sub(r'[^\w\s]', '_', filename)

def get_lyrics(
        song_url,
        save=True,
        replace=False,
        folder="songs"
):
    song = urlopen(song_url)
    soup = BeautifulSoup(song.read(), "html.parser")
    lyrics = soup.find_all("div")[22].get_text()
    title = soup.find_all("b")[1].get_text().replace('"', '')
    file_title = title.replace(" ", "_") + ".txt"  # Ensure file extension is included
    album = soup.find_all(class_="songinalbum_title")
    album_title = "dummy_title"
    if len(album) != 0:
        album_text = album[0].get_text()
        print(album_text)
        album_title = album_text.replace("album:", "").replace("compilation:", "").replace('"', "").strip()
        file_title = sanitize_filename(title.replace(" ", "_"))
        year_matches = re.findall(r'\((\d{4})\)', album_text)
        if year_matches:
            year = int(year_matches[-1])  # Take the last match in case there are multiple
        else:
            year = None
    else:
        year = None

    album_path = os.path.join(folder, album_title)  # Construct the album path

    year_passed_1999 = False
    try:
        if 1986 <= year <= 1996:
            os.makedirs(album_path, exist_ok=True)  # Ensure album directory exists
            file_path = os.path.join(album_path, file_title)
            print("File path:", file_path+'.txt')
            # Check if the file exists and replace is False
            if not os.path.exists(file_path + '.txt') or replace:
                save_file(path=file_path, text=lyrics, replace=replace)
                year_passed_1999 = False
                return year_passed_1999
            else:
                print(f"File '{file_path}' already exists. Skipping...")
                year_passed_1999 = False
                return year_passed_1999
        else:
            print(f"File '{file_title}' not in the range. Skipping...")
            if year > 1999:
                year_passed_1999 = True
            else:
                year_passed_1999 = False
            return year_passed_1999
    except TypeError:
        print(f"File '{file_title}' something is wrong with the comparison...")
        year_passed_1999 = False
        return year_passed_1999




def scrape_artist(
    az_url,
    sleep="random",
    replace=False,
    folder="songs"
):
    home = "https://www.azlyrics.com/"
    main_page = urlopen(az_url)
    bs = BeautifulSoup(main_page.read(), "html.parser")
    divs = bs.find_all('div', {"class": "listalbum-item"})
    urls = list()
    albumNames = bs.find_all('div', {"class": "album"})

    for d in divs:
        try:
            # Attempt to extract the URL
            url_part = d.a['href'].split("/", 1)[1]
            full_url = home + url_part
            urls.append(full_url)
        except TypeError:
            # If there's an error (e.g., d.a is None), this block will be executed
            print("Skipping a song link due to an error.")
            continue  # This tells the loop to move on to the next iteration
    n = len(urls)
    i = 1
    for url in urls:
        year_passed = get_lyrics(url, save=True, replace=replace, folder=folder)
        if(year_passed):
            break
        if sleep == "random":
            rt = random.randint(5, 15)
            t = 10
        else:
            rt = t = sleep
        print("Songs downloaded:", i, "/", n, " -  ETA:", round(t*(n-i)/60, 2), "minutes")
        i += 1
        time.sleep(rt)  # This is to avoid being recognized as a bot


def get_artists(
    letter,
    home="https://www.azlyrics.com/"
):
    url = home + letter + ".html"
    page = urlopen(url)
    soup = BeautifulSoup(page.read(), "html.parser")
    links = soup.find_all("div", {"class": "row"})[1].find_all("a")
    artists_urls = list()
    artists_names = list()
    for link in links:
        artists_urls.append(home + link["href"])
        artists_names.append(link.get_text())
    return artists_urls, artists_names


def scrape_all(
    letters="all",
    sleep="random",
    by_decade=True,
    replace=False,
    folder="songs"
):
    if letters == "all":
        lets = list()
        for let in ascii_lowercase:
            lets.append(let)
        lets.append(str(19))
    else:
        lets = letters
    for let in lets:
        print("---------- LETTER:", lets, "----------")
        artists_urls, artists_names = get_artists(let)
        i = 0
        for az_url in artists_urls:
            print("\n")
            print("*** NOW SCRAPING ARTIST", str(artists_names[i]), " -  (", i+1, "/", len(artists_urls), ") ***")
            if folder == "names":
                fold_name = artists_names[i]
            else:
                fold_name = folder
            scrape_artist(az_url=az_url, by_decade=by_decade, sleep=sleep, replace=replace, folder=fold_name)
            i += 1
