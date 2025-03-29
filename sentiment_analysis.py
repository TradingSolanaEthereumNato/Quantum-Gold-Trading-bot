import logging
import numpy as np
from transformers import pipeline, AutoTokenizer
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Configure logging for sentiment analysis
logger = logging.getLogger("SentimentAnalysis")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class NewsSentimentAnalyzer:
    """
    Enhanced class for performing sentiment analysis on financial news articles.
    This uses a transformer model fine-tuned for sentiment analysis.
    It also includes caching, batch processing, and sentiment confidence analysis.
    """

    def __init__(self, model: str, cache_size: int = 1000):
        """
        Initialize the NewsSentimentAnalyzer with the specified model and optional cache.
        
        :param model: The name or path to the transformer model used for sentiment analysis.
        :param cache_size: The size of the cache for storing already processed articles.
        """
        self.cache_size = cache_size
        self.cache = {}
        
        try:
            # Load the sentiment analysis pipeline from Hugging Face
            self.sentiment_pipeline = pipeline("text-classification", model=model, device=-1)
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            logger.info(f"Sentiment model {model} successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load the model: {str(e)}")
            raise ValueError(f"Failed to initialize sentiment analysis model: {str(e)}")
    
    def _cache_article(self, text: str, sentiment: str, score: float):
        """Cache the sentiment result for an article."""
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))  # Remove the oldest item to maintain cache size.
        self.cache[text] = {'sentiment': sentiment, 'score': score}
    
    def _get_cached_result(self, text: str) -> Dict[str, Any]:
        """Retrieve cached sentiment result if available."""
        return self.cache.get(text)
    
    def analyze_sentiment(self, text: str, confidence_threshold: float = 0.5) -> str:
        """
        Analyzes the sentiment of the input text using a fine-tuned transformer model.
        
        :param text: The text (news article) to analyze.
        :param confidence_threshold: Minimum confidence score for sentiment to be accepted as valid.
        :return: Sentiment result: one of ['positive', 'negative', 'neutral'].
        """
        # Check if result is cached
        cached_result = self._get_cached_result(text)
        if cached_result:
            logger.info(f"Using cached result for article: {text[:50]}...")
            return cached_result['sentiment']
        
        try:
            # Tokenize the input text for the model
            inputs = self.tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
            
            # Perform sentiment analysis
            results = self.sentiment_pipeline(inputs['input_ids'])
            
            # Extract the label and score
            sentiment_label = results[0]['label']
            sentiment_score = results[0]['score']
            
            # Map the label to a more human-friendly sentiment
            sentiment_mapping = {
                "POSITIVE": "positive",
                "NEGATIVE": "negative",
                "NEUTRAL": "neutral"
            }
            sentiment = sentiment_mapping.get(sentiment_label, "neutral")

            # Log the result with the score and confidence
            if sentiment_score >= confidence_threshold:
                logger.info(f"Sentiment analysis: {sentiment} with confidence score of {sentiment_score:.4f}")
                self._cache_article(text, sentiment, sentiment_score)
            else:
                logger.info(f"Sentiment analysis returned low confidence ({sentiment_score:.4f}), classifying as neutral")
                sentiment = "neutral"
                self._cache_article(text, sentiment, sentiment_score)
            
            return sentiment
        
        except Exception as e:
            logger.error(f"Error during sentiment analysis for article '{text[:50]}': {str(e)}")
            return "neutral"

    def analyze_batch_sentiment(self, texts: List[str], confidence_threshold: float = 0.5) -> List[str]:
        """
        Analyzes the sentiment of multiple articles (batch processing).
        
        :param texts: List of texts (news articles) to analyze.
        :param confidence_threshold: Minimum confidence score for sentiment to be accepted as valid.
        :return: List of sentiment results for each article.
        """
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(partial(self.analyze_sentiment, confidence_threshold=confidence_threshold), texts))
        return results

    def get_sentiment_scores(self, text: str) -> Dict[str, float]:
        """
        Retrieves the sentiment score and label for a single article.
        
        :param text: The article to analyze.
        :return: A dictionary containing the sentiment label and score.
        """
        result = self.analyze_sentiment(text)
        score = self.cache.get(text, {}).get('score', 0.0)
        return {'sentiment': result, 'score': score}

    def get_batch_sentiment_scores(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Retrieves sentiment scores for multiple articles.
        
        :param texts: List of articles to analyze.
        :return: List of dictionaries containing sentiment labels and scores.
        """
        return [self.get_sentiment_scores(text) for text in texts]

    def analyze_multiple_sources(self, sources: List[str], confidence_threshold: float = 0.5) -> Dict[str, List[str]]:
        """
        Analyze sentiment for articles from multiple sources in parallel.
        
        :param sources: List of sources, each containing a list of articles.
        :param confidence_threshold: Minimum confidence score for sentiment to be accepted as valid.
        :return: A dictionary with source names as keys and lists of sentiment results as values.
        """
        results = {}
        for source, articles in sources:
            results[source] = self.analyze_batch_sentiment(articles, confidence_threshold)
        return results


