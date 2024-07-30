import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_lyrics(lyrics):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(lyrics)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def main():
    # Read the CSV file
    df = pd.read_csv('fileName.csv')

    # Filter rows where LyricsStatus is True
    filtered_df = df[df['LyricsStatus'] == True]

    # Apply VADER sentiment analysis to Lyrics column
    filtered_df['Label'] = filtered_df['Lyrics'].apply(analyze_lyrics)

    # Select desired columns
    output_df = filtered_df[['Top100Year','SongTitle','Artist','LyricsStatus','Lyrics','ReleaseYear','Genre', 'Label']]

    # Save the output to a new CSV file
    output_df.to_csv('fileNameLabeled.csv', index=False)

if __name__ == "__main__":
    main()
