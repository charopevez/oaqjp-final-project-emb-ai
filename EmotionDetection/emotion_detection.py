import requests
import json

def emotion_detector(text_to_analyze):
    """
    Runs emotion detection on the given text using Watson NLP's EmotionPredict function.

    Args:
        text_to_analyze (str): The text to be analyzed for emotions.

    Returns:
        dict: A dictionary with emotion scores and the dominant emotion.
    """
    # Watson NLP EmotionPredict endpoint
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    
    # Required headers
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    
    # Input JSON payload
    input_json = {
        "raw_document": {
            "text": text_to_analyze
        }
    }
    
    # Send POST request
    response = requests.post(url, headers=headers, json=input_json)
    response.raise_for_status()
    
    # Convert response text into a dictionary
    result = json.loads(response.text)
    
    # Extract emotions from the response
    emotions = result["emotionPredictions"][0]["emotion"]
    
    # Pick the required scores
    anger = emotions["anger"]
    disgust = emotions["disgust"]
    fear = emotions["fear"]
    joy = emotions["joy"]
    sadness = emotions["sadness"]
    
    # Find the dominant emotion
    dominant_emotion = max(emotions, key=emotions.get)
    
    # Return formatted output
    return {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
        "dominant_emotion": dominant_emotion
    }