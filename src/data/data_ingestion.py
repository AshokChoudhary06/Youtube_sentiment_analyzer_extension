import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml
import logging

logger = logging.getLogger('data_ingestion') # Set the logger setup
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler() # For displaying on the terminal
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log') # For storing the errors in the files
file_handler.setLevel(logging.ERROR) # store only the error logging.Error
 
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter) # How the log will be displayed in terminal 
file_handler.setFormatter(formatter) # how the error is stored in error.log file 

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(url : str) -> pd.DataFrame:
    """ Load data from a CSV file"""
    try:
        logger.debug("Starting to load the data")
        df = pd.read_csv(url)
        logger.debug("The Data has been loaded succesfully")
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e) # %s represent a string placeholder means whatever comes should convert into string 
        raise 
    except Exception as e:
        logger.error("Unexpected error occured while loading the data : %s", e)
        raise

def preprocess_data(data : pd.DataFrame)-> pd.DataFrame:

    try:
        logger.debug("Data Preprocessing Started")

        data.dropna(inplace = True)

        data.drop_duplicates(inplace = True)

        data  = data[data['Comments'].str.len() != 1]

        data = data[data['Comments'].str.strip() != ""]

        logger.debug("Data Preprocessing Successfull")
        return data
    except KeyError as e :
        logger.error("Missing column in the data  %s ", e )
        raise
    except Exception as e :
        logger.error("Unexpected error during preprocessing %s", e)
        raise

def save_data(train_data : pd.DataFrame , test_data: pd.DataFrame , data_path : str) -> None:
    try : 
        raw_data_path = os.path.join(data_path  , 'raw')

        os.makedirs(raw_data_path, exist_ok= True)

        train_data.to_csv(os.path.join(raw_data_path , 'train.csv'), index = False)
        test_data.to_csv(os.path.join(raw_data_path , 'test.csv'), index = False)

        logger.debug("Train and Test data saved to %s", raw_data_path)
    except NotADirectoryError as e :
        logger.error("Path not found Error" , e)
        raise
    except Exception as e :
        logger.error("Unexpected error ",e)
        raise

def load_params(path: str) -> dict:
    try:
        logger.debug("Parameter loading has started")
        with open(path, 'r') as f:
            param = yaml.safe_load(f)
            logger.debug("Parameter loading Completed")
            return param
    except yaml.YAMLError as e :
        logger.error("YAML error %s", e)
        raise
    except FileNotFoundError as e :
        logger.error("File not found %s", e)
        raise
    except Exception as e :
        logger.error("Unexpected error occured %s",e)
        raise

def main():
    try:
        params = load_params(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size = params['data_ingestion']['test_size']

        #Load the data
        df = load_data('src/data/labelled_comments.xls')
        # Preprocess data
        final_df = preprocess_data(df)

        # Split the data 
        train , test = train_test_split(final_df , test_size= test_size)


        save_data(train, test, os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../data'))

    except Exception as e :
        logger.error("Failed to complete the data ingestion pipeline %s", e)

if __name__ == '__main__':
    main()