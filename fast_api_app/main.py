import joblib
from fastapi import FastAPI , Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from wordcloud import WordCloud
import mlflow
import regex as re 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import os 
import pandas as pd 
from dotenv import load_dotenv
import matplotlib.pyplot as plt 
import matplotlib 
import seaborn as sns 
matplotlib.use('Agg')
import io

load_dotenv()

CURR_DIREC = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURR_DIREC)

@asynccontextmanager
async def load_model(app: FastAPI):
    try:

        mlflow.set_tracking_uri("http://16.171.44.208:8000/")
        model_path = "models:/yt_chrome_plugin_model/24"
        app.state.model = mlflow.sklearn.load_model(model_path)
        app.state.vectorizer = joblib.load(os.path.join(ROOT_DIR,"tfidf_vectorizer.pkl"))
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
    allow_origins= ['ikogkoldmhbpecojnancbehmfhfkeckp'],
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
        stop_word_french = set(stopwords.words('french'))
        spa_stop_word = set(stopwords.words('spanish'))
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

class CommentData(BaseModel):
    text : str
    date : str

class TimelineRequestModel(BaseModel):
    comment : list[CommentData]

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


        prediction = app.state.model.predict(vectorized_comments.toarray()).tolist()

        prediction = [str(pred) for pred in prediction]
        total_comments = len(data.comment)
        
        # 1. Average Length (analyzing the statistical distribution of viewer effort)
        avg_length = round(sum(len(str(c).split()) for c in data.comment) / total_comments) if total_comments > 0 else 0
        
        # 2. Sentiment Ratio
        pos_count = prediction.count("positive") # Ensure case matches your model output
        neg_count = prediction.count("negative")
        sentiment_ratio = round(pos_count / neg_count, 1) if neg_count > 0 else pos_count
        response = {
            "predictions": [{"comment": c, "sentiment": s} for c, s in zip(data.comment, prediction)],
            "kpis": {
                "total": total_comments,
                "avg_words": avg_length,
                "ratio": sentiment_ratio
            }
        }
        return response


    except Exception as e:
        return {"error": f"Prediction failed due to {e}"}


@app.post('/generate_wordcloud')
def gen_wordcloud(data: Cleancomment):
    try:
        if not data.comment:
            return {"error":"No comment data is available"}
        
        clean_comment = " ".join([preprocess_commmet(c) for c in data.comment if c.strip()])

        wc = WordCloud(
            width= 600, 
            height = 300, 
            background_color= "white",
            colormap= "viridis",
            max_words= 80,
            collocations= False
        ).generate(clean_comment)

        img_by = io.BytesIO()
        fig, ax = plt.subplots(figsize=(6,3))
        ax.imshow(wc, interpolation = 'bilinear')
        ax.axis('off')
        fig.tight_layout(pad = 0)
        fig.savefig(img_by, format = 'png' , bbox_inches = 'tight')
        plt.close(fig) # Explicitly close THIS specific figure

        img_by.seek(0)

        return StreamingResponse(img_by , media_type= 'image/png')
    except Exception as e:
        return f"Failed to load the word cloud {e}"

@app.post('/generate_heatmap')
def gen_heatmap(data : TimelineRequestModel):
    try: 
        if not data.comment:
            return {"error":"No comment data avaialble"}

        text = [c.text for c in data.comment]
        date = [c.date for c in data.comment]

        clean_data = [preprocess_commmet(c) for c in text]
        vectorize = app.state.vectorizer.transform(clean_data)
        feature_names = app.state.vectorizer.get_feature_names_out()

        vectorized_df = pd.DataFrame(
                    vectorize.toarray(),
                    columns = feature_names
                )
        prediction = app.state.model.predict(vectorized_df)

        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(date),
                "Sentiment" : prediction 
            }
        )

        df['Month'] = df['Date'].dt.strftime('%Y-%b')

        #3. Calculate Percentages per Month
        # Count the occurrences of each sentiment per month
        monthly_counts = df.groupby(['Month', 'Sentiment']).size().unstack(fill_value=0)
        
        # Convert raw counts to percentages (row-wise)
        heatmap_data = monthly_counts.div(monthly_counts.sum(axis=1), axis=0) * 100

        for col in ['positive', 'neutral', 'negative']:
            if col not in heatmap_data.columns:
                heatmap_data[col] = 0.0

        heatmap_data = heatmap_data[['positive', 'neutral', 'negative']]

        heatmap_data.index = pd.to_datetime(heatmap_data.index, format='%Y-%b')
        heatmap_data = heatmap_data.sort_index()
        heatmap_data.index = heatmap_data.index.strftime('%b %y') # E.g., 'Jan 23'

        fig, ax = plt.subplots(figsize=(6, len(heatmap_data) * 0.4 + 1))
        
        sns.heatmap(
            heatmap_data, 
            annot=True,          
            fmt=".0f",           
            cmap="Blues",        
            cbar=False,          
            linewidths=0.5,      
            annot_kws={"size": 10},
            ax=ax # Force Seaborn to draw exactly on this isolated axis
        )

        ax.set_title("Sentiment Timeline (%)", pad=10, fontsize=12)
        ax.set_ylabel("") 
        ax.set_xlabel("") 
        fig.tight_layout()

        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close(fig) # Explicitly close THIS specific figure
        
        img_buffer.seek(0)
        return StreamingResponse(img_buffer, media_type="image/png")
    
    except Exception as e:
        return {"error": f"Heatmap generation failed: {str(e)}"}