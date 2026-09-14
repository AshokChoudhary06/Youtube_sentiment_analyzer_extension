import pandas as pd
import os 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import logging
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score 

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

file_handler = logging.FileHandler('model_build.log')
file_handler.setLevel(logging.ERROR)

format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(format)
file_handler.setFormatter(format)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(url):
    try:
        train_data = pd.read_csv(url)
        logger.debug(f"Training Data has been loadaed from {url}")
        return train_data 
    
    except Exception as e : 
        logger.error(f"Failed to load the data", e)
        raise

def load_params(path :str):
    try:
        with open(path , 'r') as f:
            param = yaml.safe_load(f)
            return param 
    except yaml.YAMLError as e :
        logger.error(f"Parameter Loading failed" , e)
        raise
    except Exception as e :
        logger.error(f"Unexpected error occured", e)
        raise 

def vectorize_data(train_data, ngram_range , max_feature, sublinear_tf):
    try: 

        vectorizer = TfidfVectorizer(ngram_range = ngram_range , max_features = max_feature, sublinear_tf= sublinear_tf)
        x_train_tfidf  = vectorizer.fit_transform(train_data)

          # Save the vectorizer in the root directory
        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug('TF-IDF applied with trigrams and data transformed')
        return x_train_tfidf 
    except Exception as e :
        logger.error(f"Failed to vectorize the data {e}")
        raise

def model_training(x_train_transform , y_train, c, l1_ratio , class_weight):
    try :
        model = LogisticRegression(C = c , l1_ratio= l1_ratio , class_weight= None)
        model.fit(x_train_transform, y_train)
        return model
    
    except Exception as e:
        logger.error(f"Failed to train the model {e}")
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def main():
    try:

        root_dir = get_root_directory()
        param = load_params(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
    
        ngram_range = tuple(param['model_building']['ngram_range'])
        max_feature = param['model_building']['max_feature']
        sublinear_tf = param['model_building']['sublinear_tf']

        c = param['model_building']['C']
        l1_ratio = param['model_building']['l1_ratio']
        class_weight = param['model_building']['class_weight']


        data = load_data("data/interim/preprocessed_train.csv")
        data = data.dropna(subset = ['Comments', 'Sentiment']).reset_index(drop = True)
        x =data['Comments']
        y = data['Sentiment']
        x_train_tfidf = vectorize_data( x, ngram_range , max_feature , sublinear_tf)

        best_model = model_training(x_train_tfidf , y , c , l1_ratio , class_weight)
        save_model(best_model ,  os.path.join(root_dir, 'log_model.pkl'))

    except Exception as e:
        logger.error(f"Unexpected error occured {e}")
        raise 

if __name__ == '__main__':
    main()