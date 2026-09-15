import joblib
from fastapi import FastAPI , Request
from pydantic import BaseModel
import mlflow
import regex as re 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import os 
import pandas as pd 
from dotenv import load_dotenv

load_dotenv()


os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") # Modify the operating system envrionment variables 
os.environ["AWS_DEFAULT_REGION"]  = os.getenv("AWS_DEFAULT_REGION")

@asynccontextmanager
async def load_model(app: FastAPI):
    try:

        mlflow.set_tracking_uri("http://16.171.44.208:8000/")
        model_path = f"models:/yt_chrome_plugin_model/1"
        app.state.model = mlflow.pyfunc.load_model(model_path)
        app.state.vectorizer = joblib.load("../tfidf_vectorizer.pkl")
        yield
        print("Shutting the server and clearing memory")
        del app.state.model
        del app.state.vectorizer

    except FileNotFoundError as e:
        print("File not found at the specified path:", e)
        raise
    except Exception as e:
        print("Unable to load the model due to some unexpected error:",e)
        raise

app = FastAPI(lifespan= load_model)
app.add_middleware(
    CORSMiddleware,
    allow_origins= ['*'],
    allow_headers = ['*'],
    allow_credentials = ['*'],
    allow_methods = ['*']
)

def preprocess_commmet(comment: str):
    """Apply the preprocessing to the comments"""
    try:
        comment = comment.lower() 

        comment = comment.strip()

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
        print("Error in prerpocessing the comments", e)
        return comment 


class Cleancomment(BaseModel):
    comment : list[str] 

@app.get("/")
def test():
    return "Working"

@app.post("/predict")
def predict(data : Cleancomment):
    try:
        if not data.comment:
            return {"error":"Data not received"}

        cleaned_comments = [preprocess_commmet(c) for c in data.comment]

        vectorized_comments  = app.state.vectorizer.transform(cleaned_comments)
        feature_names = app.state.vectorizer.get_feature_names_out()

        vectorized_df = pd.DataFrame(
            vectorized_comments.toarray(),
            columns = feature_names
        )

        prediction = app.state.model.predict(vectorized_df).tolist()

        prediction = [str(pred) for pred in prediction]
        response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(data.comment, prediction)]
        return response

    except Exception as e:
        return {"error": f"Prediction failed due to {e}"}