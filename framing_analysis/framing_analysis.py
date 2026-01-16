"""
Comparative framing analysis using BERTopic and class-based TF-IDF.

Analyzes how government and opposition parties frame identical policy issues
using different rhetoric through semantic topic modeling.
"""

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')


class FramingAnalyzer:
    """Semantic topic modeling with class-based TF-IDF comparison."""
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2", language="english"):
        self.sentence_model = SentenceTransformer(embedding_model)
        self.language = language
        self.topic_model = None
        
    def fit(self, texts, verbose=True):
        """Fit BERTopic model on speech texts."""
        if verbose:
            print(f"Encoding {len(texts)} texts...")
        embeddings = self.sentence_model.encode(texts, show_progress_bar=verbose)
        
        vectorizer = CountVectorizer(stop_words=self.language, 
                                    min_df=2, ngram_range=(1, 2))
        
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            vectorizer_model=vectorizer,
            verbose=verbose
        )
        
        topics, _ = self.topic_model.fit_transform(texts, embeddings)
        if verbose:
            n_topics = len(set(topics)) - 1
            print(f"Identified {n_topics} topics")
        
        return self
    
    def compare_framing(self, texts, classes, class_names=None, top_n=15):
        """Compare distinctive terms between classes for each topic."""
        if self.topic_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        topics, _ = self.topic_model.transform(texts)
        
        df = pd.DataFrame({'text': texts, 'topic': topics, 'class': classes})
        df = df[df['topic'] != -1]  # Remove outliers
        
        if class_names is None:
            class_names = sorted(df['class'].unique())
        
        results = []
        for topic_id in sorted(df['topic'].unique()):
            topic_df = df[df['topic'] == topic_id]
            
            if len(topic_df) < 10:
                continue
            
            # Topic label
            topic_words = self.topic_model.get_topic(topic_id)
            topic_label = ', '.join([w for w, _ in topic_words[:5]]) if topic_words else f"Topic {topic_id}"
            
            # Class counts
            class_counts = topic_df['class'].value_counts()
            
            result = {'topic_id': topic_id, 'topic_label': topic_label, 'n_speeches': len(topic_df)}
            
            # Extract terms per class
            for class_name in class_names:
                class_texts = topic_df[topic_df['class'] == class_name]['text'].tolist()
                result[f'{class_name}_count'] = class_counts.get(class_name, 0)
                
                if len(class_texts) >= 5:
                    vectorizer = CountVectorizer(stop_words=self.language, min_df=2, ngram_range=(1, 2))
                    X = vectorizer.fit_transform(class_texts)
                    words = vectorizer.get_feature_names_out()
                    scores = X.toarray().sum(axis=0)
                    top_words = sorted(zip(words, scores), key=lambda x: x[1], reverse=True)
                    result[f'{class_name}_terms'] = ', '.join([w for w, _ in top_words[:top_n]])
                else:
                    result[f'{class_name}_terms'] = ''
            
            results.append(result)
        
        return pd.DataFrame(results)
