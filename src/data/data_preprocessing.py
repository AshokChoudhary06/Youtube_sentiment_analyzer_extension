import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
import nltk
import os  
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging 
import regex as re 
nltk.download('stopwords')
nltk.download('wordnet')

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('preprocess_error.log')
file_handler.setLevel(logging.ERROR)

format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(format)
file_handler.setFormatter(format)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def preprocess_data(comment):
    try:
        comment = comment.lower() 

        comment  = re.sub(r'\n',' ', comment)

        en_stop_words = set(stopwords.words('english'))
        stop_word_french = set(stopwords.words('French'))
        spa_stop_word = set(stopwords.words('Spanish'))
        arab_stop_word = set(stopwords.words('arabic'))
        bengal_stop_word = set(stopwords.words('bengali'))
        russ_stop_word = set(stopwords.words('russian'))
        turk_stop_word = set(stopwords.words('turkish'))
        hinglish_stop_word = {
            "hai", "ki", "ho", "ka", "ke", "ek", "mein", "se", "ko", "aur", 
            "ye", "yeh", "woh", "tha", "thi", "hi", "kya", "toh", "ne", "ha", "na" }
        global_stop_word = en_stop_words.union(stop_word_french, spa_stop_word, hinglish_stop_word, arab_stop_word, bengal_stop_word, russ_stop_word, turk_stop_word)
        stop_word_mod = set(global_stop_word) -  {'not', 'but', 'however', 'no', 'yet'}

        comment = ' '.join([word for word in comment.split() if word not in stop_word_mod]) # remove the stopwords

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment 
    except Exception as e:
        logger.error('Error in processing comment: ', e)
        return comment 

def  normalize_text(df):
    """Apply preprocessing to the text """
    try:
       logger.debug("Preprocessing the text started")
       df['Comments'] = df['Comments'].apply(preprocess_data)
       return df 

    except Exception as e :
        logger.error(f"Text Normaliztion failed: {e}")
        raise

def save_data(train_data, test_data , data_path : str):
    try: 
        interim_data_path = os.path.join(data_path , 'interim')
        logger.debug(f'Creating Directory at {interim_data_path}')

        os.makedirs(interim_data_path , exist_ok= True)
        logger.debug(f"Directory created at {interim_data_path}")

        train_data.to_csv(os.path.join(interim_data_path, 'preprocessed_train.csv'), index = False)
        test_data.to_csv(os.path.join(interim_data_path, 'preprocessed_test.csv'), index = False)

        logger.debug(f"Preprocessed data saved to the {interim_data_path}")
    except Exception as e :
        logger.error(f"Failed to save the preprocessed data {e}")
        raise 

def main():

    try:
        train_data = pd.read_csv('data/raw/train.csv')
        test_data= pd.read_csv('data/raw/test.csv')

        normalize_train_data = normalize_text(train_data)
        normalize_test_data = normalize_text(test_data)

        save_data(normalize_train_data , normalize_test_data, data_path = './data')

    except Exception as e:
        logger.error(f"Error occured while preprocessing the data {e}")

if __name__ == '__main__':
    main()