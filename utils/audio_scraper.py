import os
import pandas as pd
import yt_dlp 

audio_dir = 'audio'
east_coast_west_coast = 'east_coast'
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

# Need to store the data in a dataframe like this
songs_to_scrape = pd.read_csv('songs_to_scrape.csv')

for i, r in songs_to_scrape.iterrows():
    destination_dir = os.path.join(audio_dir, east_coast_west_coast)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    youtube_url = r['youtube_url']
    if youtube_url:
        fn = '{}_{}'.format(r['song'], r['performer'])

        # Skip if file already exists
        fp = os.path.join(destination_dir, fn)
        if not os.path.exists(fp + '.mp3'):

            ydl_opts = {
                'outtmpl': fp + '.%(ext)s',
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '320',
                }],
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
            except Exception as e:
                print(e)
                pass
        else:
            print('{} already exists'.format(fp))