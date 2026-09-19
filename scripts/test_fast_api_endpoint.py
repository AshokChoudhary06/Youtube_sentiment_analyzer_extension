import pytest
from fastapi.testclient import TestClient
from fast_api_app.main import app

client = TestClient(app)

def test_predict_endpoint():
    data  = {
        "comment" :["These is the good video", "I hate watching samuel and her together"]
    }

    response = client.post(f"/predict" , json = data)

    assert response.status_code == 200 , "Api failed to process the prediction"
    assert isinstance(response.json(),  list)

def test_generate_cloud():
    data = {
        "comment": ["Love this!", "Not so great.", "Absolutely amazing!", "Horrible experience."]
    }
    response = client.post(f"/generate_wordcloud", json=data)
    assert response.status_code == 200 , "Wordcloud generation Failed"
    assert response.headers["Content-Type"] == "image/png", "The response is not the png file"

def test_generate_heat_map():
    payload = {
        "comment": [
            {"text": "This video is amazing!", "date": "2023-10-01T12:00:00Z"},
            {"text": "I did not like the audio.", "date": "2023-11-05T15:30:00Z"},
            {"text": "Very helpful tutorial.", "date": "2023-11-10T09:15:00Z"}
        ]
    }
    response = client.post(f"/generate_heatmap", json= payload)
    assert response.status_code == 200  , "heat_map generation failed"
    assert response.headers["Content-Type"] == "image/png" , "the response is not png file"