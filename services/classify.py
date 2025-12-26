import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Single ton class to classify 
class Classify:
    __instance = None
    def __new__(cls, model_name="cardiffnlp/twitter-roberta-base-sentiment"):
        if cls.__instance is None:
            # New instance if class does not exist
            cls.__instance = super(Classify, cls).__new__(cls)
            cls.__instance.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Initial Roberta Sentiment model  to inference mode
            cls.__instance.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            cls.__instance.model.eval()
            cls.__instance.labels = ['negative', 'neurtal', 'positive']

        # Return class each time
        return cls.__instance
    
    # Prediction method
    def predict(self, texts, max_length):
        try:
            # Transfrom text for model compatible input
            input_tokens = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            # Run model in inference mode
            print(f"Predicting text sentiments...")
            with torch.inference_mode():
                outputs = self.model(**input_tokens)
            
            # Convert outputs to labels for texts
            # --> Normalize, softmax
            output_probabilites = torch.softmax(outputs, dim=1)
            # --> Get argmax off probabilities
            output_predictions = torch.argmax(output_probabilites, dim=1)

            # Get classification results
            results = []
            for text, pred_id, prob_row in zip(texts, output_predictions, output_probabilites):
                results.append({
                    "text" : text,
                    "lable": self.label[pred_id],
                    "confidence": prob_row[pred_id].item()
                })
            
            return results
        except Exception as e:
            print(f"Error while classifying texts", e)
            return []