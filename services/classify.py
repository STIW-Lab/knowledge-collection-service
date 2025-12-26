import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Single ton class to classify 
class Classify:
    __instance = None
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            # New instance if class does not exist
            cls.__instance = super(Classify, cls).__new__(cls, *args, **kwargs)
        # Return class each time
        return cls.__instance
    
    def __init__(self):
        pass