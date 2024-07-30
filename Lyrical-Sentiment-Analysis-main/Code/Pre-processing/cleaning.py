import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
import pandas as pd
import random
import contractions

'''# Read the CSV file into a pandas DataFrame
df = pd.read_csv('Song_Lyrics_For_Top_100_Songs_2014_to_2023.csv')

# get column names
colNames = ['Top100Year','SongTitle','Artist','LyricsStatus','Lyrics','ReleaseYear','Genre']

# Filter rows where LyricsStatus is True
filtered_df = df[df['LyricsStatus'] == 'TRUE']

# Select desired columns
output_df = filtered_df[['Top100Year','SongTitle','Artist','LyricsStatus','Lyrics','ReleaseYear','Genre']]

# Save the output to a new CSV file
output_df.to_csv('cleanedData.csv', index=False)'''
def split_by_spaces(lyrics):
    return lyrics.split()

def remove_commas(lyrics):
    sentence = ''.join(lyrics)
    fixed = sentence.replace(',', '')
    return fixed.split()

def remove_parenthesis(lyrics):
    filtered = [word for word in lyrics if not (word.startswith('(') and word.endswith(')'))]
    return filtered

def fix_contractions(lyrics):
    sentence = ' '.join(lyrics)
    fixed = contractions.fix(sentence)
    return fixed.split()

def fix_spelling(spell, lyrics):
    misspelled = spell.unknown(lyrics)
    fixed = [spell.correction(word) if word in misspelled else word for word in words]
    fixed = [x for x in fixed if x is not None]
    return fixed



if __name__ == "__main__":
    stop_words = set(stopwords.words('english'))
    negation_words = set(['not', 'no', 'never', 'none', 'nobody', 'nowhere', 'nothing'])
    spell = SpellChecker()

    # using this array to simulate having multiple lyrics to process
    lyrics_array = ["I'm not happi and that's not good (good), it's not good eno-o-ogh"]

    for lyrics in lyrics_array:
        print(f'original: {lyrics}')

        words = lyrics.split()

        # Remove commas
        words = remove_commas(lyrics)
        print(f'commas: {words}')

        # Remove anything in parenthesis
        words = remove_parenthesis(words)
        print(f'parenthesis: {words}')

        # Fix contractions
        words = fix_contractions(words)
        print(f'contractions: {words}')

        # Spellcheck replacement
        words = fix_spelling(spell, words)   
        print(f'spelling: {words}')

        # Remove stop words
        words = [word for word in words if word.lower() not in stop_words or word.lower() in negation_words]
        print(f'stop words: {words}')

        filtered_text = ' '.join(words)
        print(f'final: {filtered_text}')