"""
Emotion analysis using DistilRoBERTa for discrete emotion classification.

Classifies speeches into six emotion categories and compares patterns
across government and opposition parties.
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')


class EmotionAnalyzer:
    """Discrete emotion classification across 6 categories."""
    
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading emotion model on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    
    def predict(self, texts, batch_size=16):
        """Predict emotion probabilities for texts."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                   max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            for prob in probs:
                prob_dict = {label: prob[idx].item() 
                           for idx, label in enumerate(self.emotion_labels)}
                predicted = max(prob_dict, key=prob_dict.get)
                results.append({'predicted_emotion': predicted, 'confidence': prob_dict[predicted], **prob_dict})
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)}")
        
        return results
    
    def analyze_dataframe(self, df, text_col='text', batch_size=16):
        """Add emotion predictions to dataframe."""
        texts = df[text_col].fillna("").tolist()
        results = self.predict(texts, batch_size)
        emotion_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), emotion_df], axis=1)


def compare_emotions(df, group_col='speaker_type', groups=['Government', 'Opposition']):
    """Compare emotion distributions between groups."""
    emotion_cols = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    
    results = []
    for group in groups:
        group_df = df[df[group_col] == group]
        if len(group_df) == 0:
            continue
        
        result = {'group': group, 'n_speeches': len(group_df)}
        for emotion in emotion_cols:
            result[f'{emotion}_mean'] = group_df[emotion].mean()
        
        results.append(result)
    
    return pd.DataFrame(results)
