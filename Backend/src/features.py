"""
Feature extraction utilities.
Implements TF-IDF vectorization, POS tag ratios, negation counts, pronoun counts, and emotion keywords.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from collections import Counter
import string

# Download required NLTK data (will fail silently if already downloaded)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass


# Emotion/affect keywords
NEGATIVE_EMOTION_KEYWORDS = {
    'fear', 'afraid', 'scared', 'terrified', 'anxious', 'worried', 'panic',
    'anger', 'angry', 'mad', 'furious', 'rage', 'hostile', 'hate',
    'sad', 'sadness', 'depressed', 'down', 'melancholy', 'gloomy',
    'stress', 'stressed', 'pressure', 'tense', 'nervous', 'uneasy'
}

POSITIVE_EMOTION_KEYWORDS = {
    'happy', 'happiness', 'joy', 'joyful', 'glad', 'pleased', 'delighted',
    'excited', 'enthusiastic', 'thrilled', 'eager', 'energetic',
    'calm', 'peaceful', 'relaxed', 'content', 'satisfied', 'comfortable',
    'love', 'loved', 'loving', 'affection', 'warm', 'kind', 'caring'
}

# Negation words
NEGATION_WORDS = {
    'not', 'no', 'never', 'nothing', 'none', 'nowhere', 'nobody', 'noone',
    "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't", "can't",
    "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't"
}

# Pronouns
PRONOUNS = {
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves'
}


class FeatureExtractor:
    """Extract various linguistic features from text."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def count_negations(self, text: str) -> int:
        """Count negation words in text."""
        tokens = word_tokenize(text.lower())
        return sum(1 for token in tokens if token in NEGATION_WORDS)
    
    def count_pronouns(self, text: str) -> int:
        """Count pronouns in text."""
        tokens = word_tokenize(text.lower())
        return sum(1 for token in tokens if token in PRONOUNS)
    
    def count_emotion_keywords(self, text: str) -> dict:
        """Count positive and negative emotion keywords."""
        tokens = word_tokenize(text.lower())
        token_set = set(tokens)
        
        pos_count = len(token_set & POSITIVE_EMOTION_KEYWORDS)
        neg_count = len(token_set & NEGATIVE_EMOTION_KEYWORDS)
        
        return {
            'positive_emotions': pos_count,
            'negative_emotions': neg_count,
            'emotion_ratio': pos_count / (neg_count + 1)  # +1 to avoid division by zero
        }
    
    def get_pos_ratios(self, text: str) -> dict:
        """Calculate POS tag ratios."""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Count POS tags
            tag_counts = Counter(tag for word, tag in pos_tags)
            total = len(tokens)
            
            if total == 0:
                return {
                    'noun_ratio': 0.0,
                    'verb_ratio': 0.0,
                    'adj_ratio': 0.0,
                    'adv_ratio': 0.0,
                    'pronoun_ratio': 0.0
                }
            
            # POS tag categories
            nouns = sum(tag_counts[tag] for tag in tag_counts.keys() 
                       if tag.startswith('NN'))
            verbs = sum(tag_counts[tag] for tag in tag_counts.keys() 
                       if tag.startswith('VB'))
            adjectives = sum(tag_counts[tag] for tag in tag_counts.keys() 
                            if tag.startswith('JJ'))
            adverbs = sum(tag_counts[tag] for tag in tag_counts.keys() 
                         if tag.startswith('RB'))
            pronouns = sum(tag_counts[tag] for tag in tag_counts.keys() 
                          if tag.startswith('PRP'))
            
            return {
                'noun_ratio': nouns / total,
                'verb_ratio': verbs / total,
                'adj_ratio': adjectives / total,
                'adv_ratio': adverbs / total,
                'pronoun_ratio': pronouns / total
            }
        except:
            return {
                'noun_ratio': 0.0,
                'verb_ratio': 0.0,
                'adj_ratio': 0.0,
                'adv_ratio': 0.0,
                'pronoun_ratio': 0.0
            }
    
    def extract_all_features(self, text: str) -> dict:
        """Extract all features from a single text."""
        features = {}
        
        # Negation count
        features['negation_count'] = self.count_negations(text)
        
        # Pronoun count
        features['pronoun_count'] = self.count_pronouns(text)
        
        # Emotion keywords
        emotion_features = self.count_emotion_keywords(text)
        features.update(emotion_features)
        
        # POS tag ratios
        pos_features = self.get_pos_ratios(text)
        features.update(pos_features)
        
        # Text length features
        features['char_count'] = len(text)
        features['word_count'] = len(word_tokenize(text))
        features['avg_word_length'] = features['char_count'] / (features['word_count'] + 1)
        
        return features
    
    def extract_batch_features(self, texts) -> list:
        """Extract features from a batch of texts."""
        return [self.extract_all_features(text) for text in texts]


def create_tfidf_vectorizer(max_features: int = 5000, ngram_range: tuple = (1, 2)):
    """
    Create a TF-IDF vectorizer.
    
    Args:
        max_features: Maximum number of features
        ngram_range: Range of n-grams to consider
        
    Returns:
        TfidfVectorizer instance
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        lowercase=True,
        min_df=2,
        max_df=0.95
    )


if __name__ == "__main__":
    # Test feature extraction
    extractor = FeatureExtractor()
    
    test_text = "I'm not happy about this situation. I feel very worried and anxious."
    
    features = extractor.extract_all_features(test_text)
    print("Features extracted:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    # Test TF-IDF
    vectorizer = create_tfidf_vectorizer(max_features=100)
    test_texts = ["This is a test sentence.", "Another test sentence here."]
    X = vectorizer.fit_transform(test_texts)
    print(f"\nTF-IDF matrix shape: {X.shape}")

