import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Single ton class to classify 
class Classify:
    __instance = None
    def __new__(cls, model_name="cardiffnlp/twitter-roberta-base-sentiment", *args, **kwargs):
        if cls.__instance is None:
            # New instance if class does not exist
            cls.__instance = super(Classify, cls).__new__(cls, model_name, *args, **kwargs)
            cls.__instance.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Initial Roberta Sentiment model  to inference mode
            cls.__instance.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            cls.__instance.model.eval()
            cls.__instance.labels = ['negative', 'nuertal', 'positive']
            
        # Return class each time
        return cls.__instance
    
    def __init__(self):
        pass