# PopMIR

Learn more about the project [here.](https://www.jku.at/en/institute-of-computational-perception/research/projects/popmir/) 

### Initial Data Buildup & Lyrics Scraping

Our data comes from the https://www.rollingstone.com/ website because it has a good reputation in the music industry.
We used the [Top 100 ranked East-Coast](https://www.rollingstone.com/music/music-lists/best-east-coast-rap-songs-1234737704/o-p-p-naughty-by-nature-1234737749/) and [Top 100 rated West-Coast](https://www.rollingstone.com/music/music-lists/best-west-coast-hip-hop-songs-1234712968/) songs as a baseline for our database.
We scraped the data and saved it in the following format in a .json file:

```json
{
  "ranked number on rollingstone": {
    "title": "Song title",
    "artist": "Song artist",
    "lyrics": "Song lyrics",
    "youtube_url": "Song URL on Youtube"
  }
}
```

During the lyrics scraping process we used a package called [lyricsgenius](https://lyricsgenius.readthedocs.io). 
For the documentation please see the attached link.
For the code please see the **utils/lyrics_scraper.ipynb** file.

### Audio Scraping

With the Youtube URLs, acquired in the previous step and with the help of a package called [ytb-dlt](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjUgujrlv2CAxWAQEEAHT6oBRwQFnoECBYQAQ&url=https%3A%2F%2Fgithub.com%2Fyt-dlp%2Fyt-dlp&usg=AOvVaw2bas9zcrYkd2gTMn6wTaVh&opi=89978449) we were able to scrape down the audio files.
For the code please see **utils/audio_scraper.py**