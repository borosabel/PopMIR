# PopMIR

Learn more about the project [here.](https://www.jku.at/en/institute-of-computational-perception/research/projects/popmir/) 

### Initial Data Buildup

Our data comes from the https://www.rollingstone.com/ website because it has a good reputation in the music industry.
We used the Top 100 ranked East-Coast and Top 100 rated West-Coast songs as a baseline for our database.
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

For the code please see the **lyrics_scraper.ipynb** file.