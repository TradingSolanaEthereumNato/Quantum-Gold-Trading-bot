import asyncio
import logging
import httpx
import os
from typing import List, Dict, Any, Optional
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.deberta_v2 import DebertaV2Tokenizer  # Critical fix
from sentence_transformers import SentenceTransformer
from utils import quantum_entangled_database  # Custom module

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Constants
SENTIMENT_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
CROSS_ENCODER_MODEL = "cross-encoder/nli-deberta-v3-base"
EMBEDDER_MODEL = "sentence-transformers/all-mpnet-base-v2"
MAX_ARTICLE_LENGTH = 768
GPT4_TIMEOUT = 15
RISK_KEYWORDS = {
    "crash", "scam", "hack", "collapse", "sanction",
    "investigation", "warning", "alert", "fraud", "manipulation"
}

# News Sources Configuration
QUANTUM_SOURCES = {
    "newsapi": {
        "url": "https://newsapi.org/v2/everything",
        "params": {
            "apiKey": os.getenv("NEWS_API_KEY"),
            "q": "(gold OR XAUUSD OR XAUAUD) AND (forecast OR analysis)",
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 20
        }
    },
    "reuters_gold_rss": "http://feeds.reuters.com/reuters/commodities",
    "bloomberg_metal_rss": "https://news.google.com/rss/search?q=gold+OR+XAUUSD+site:bloomberg.com&hl=en-US&gl=US&ceid=US:en",
    "kitco_news": "https://www.kitco.com/rss/news.xml",
    "goldprice_rss": "https://www.goldprice.org/rss.xml",
    "tradingview_analysis": "https://www.tradingview.com/feed/rss/?stream=gold"
}

FUNDAMENTAL_SOURCES = {
    "central_banks_rss": "https://www.bis.org/rss/cbanks.htm",
    "economic_calendar_api": {
        "url": "https://economic-calendar-api.p.rapidapi.com/news",
        "params": {"apiKey": os.getenv("ECON_CALENDAR_KEY")}
    },
    "bloomberg_economics": "https://www.bloomberg.com/markets/economics"
}

GEOPOLITICAL_SOURCES = {
    "global_conflict_monitor": "https://acleddata.com/rss/",
    "diplomacy_rss": "https://www.cfr.org/rss/global",
    "sanctions_api": {
        "url": "https://sanctions-api.p.rapidapi.com/news",
        "params": {"apiKey": os.getenv("SANCTIONS_API_KEY")}
    }
}

class FundamentalDataAPI:
    """Handles fundamental data news (economic indicators, central bank decisions, etc.)"""

    def __init__(self):
        self.sentiment_analyzer = NewsSentimentAnalyzer()

    async def fetch_fundamental_news(self) -> List[Dict]:
        """Fetch and process fundamental economic news."""
        try:
            raw_data = await self._fetch_category_news(FUNDAMENTAL_SOURCES)
            processed_data = await self._process_fundamental_news(raw_data)
            return processed_data
        except Exception as e:
            logger.error(f"Error fetching fundamental news: {str(e)}")
            return []

    async def _fetch_category_news(self, sources: Dict) -> List[Dict]:
        """Fetch news from category-specific sources."""
        results = []
        for source_name, source_config in sources.items():
            try:
                if isinstance(source_config, dict):  # API endpoint
                    async with httpx.AsyncClient() as client:
                        response = await client.get(source_config["url"], params=source_config["params"])
                        if response.status_code == 200:
                            results.extend(response.json().get("articles", []))
                else:  # RSS feed
                    # Placeholder for RSS implementation
                    pass
            except Exception as e:
                logger.warning(f"Failed to fetch from {source_name}: {str(e)}")
        return results

    async def _process_fundamental_news(self, raw_data: List[Dict]) -> List[Dict]:
        """Process raw fundamental news data."""
        processed = []
        for article in raw_data:
            sentiment = self.sentiment_analyzer.analyze_sentiment(article.get("content", ""))
            processed.append({
                "title": article.get("title", ""),
                "content": article.get("content", ""),
                "source": article.get("source", {}).get("name", "unknown"),
                "sentiment": sentiment,
                "category": "fundamental"
            })
        return processed

class GeopoliticalRiskAPI:
    """Handles geopolitical risk analysis and news monitoring."""

    def __init__(self):
        self.sentiment_analyzer = NewsSentimentAnalyzer()

    async def fetch_geopolitical_risks(self) -> List[Dict]:
        """Fetch and analyze geopolitical risk events."""
        try:
            raw_data = await self._fetch_category_news(GEOPOLITICAL_SOURCES)
            processed_data = await self._process_geopolitical_news(raw_data)
            return processed_data
        except Exception as e:
            logger.error(f"Error fetching geopolitical risks: {str(e)}")
            return []

    async def _fetch_category_news(self, sources: Dict) -> List[Dict]:
        """Fetch news from category-specific sources."""
        results = []
        for source_name, source_config in sources.items():
            try:
                if isinstance(source_config, dict):  # API endpoint
                    async with httpx.AsyncClient() as client:
                        response = await client.get(source_config["url"], params=source_config["params"])
                        if response.status_code == 200:
                            results.extend(response.json().get("articles", []))
                else:  # RSS feed
                    # Placeholder for RSS implementation
                    pass
            except Exception as e:
                logger.warning(f"Failed to fetch from {source_name}: {str(e)}")
        return results

    async def _process_geopolitical_news(self, raw_data: List[Dict]) -> List[Dict]:
        """Process raw geopolitical news data."""
        processed = []
        for article in raw_data:
            sentiment = self.sentiment_analyzer.analyze_sentiment(article.get("content", ""))
            processed.append({
                "title": article.get("title", ""),
                "content": article.get("content", ""),
                "source": article.get("source", {}).get("name", "unknown"),
                "sentiment": sentiment,
                "category": "geopolitical"
            })
        return processed

class NewsSentimentAnalyzer:
    """Handles news sentiment analysis using transformer models with DeBERTa fix."""

    def __init__(self):
        # Initialize sentiment analysis components
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
        
        # Initialize cross-encoder with explicit DeBERTa tokenizer (FIXED)
        self.cross_encoder_tokenizer = DebertaV2Tokenizer.from_pretrained(
            CROSS_ENCODER_MODEL,
            use_fast=False  # Force slow tokenizer for compatibility
        )
        self.cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_MODEL)
        
        # Create pipelines
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model=self.sentiment_model,
            tokenizer=self.sentiment_tokenizer,
            device=-1  # CPU
        )
        
        self.cross_encoder = pipeline(
            "text-classification",
            model=self.cross_encoder_model,
            tokenizer=self.cross_encoder_tokenizer,
            device=-1  # CPU
        )
        
        # Initialize other components
        self.embedder = SentenceTransformer(EMBEDDER_MODEL)
        self.news_cache = self._initialize_cache()

    def _initialize_cache(self) -> Dict:
        """Initialize the news cache."""
        try:
            if hasattr(quantum_entangled_database, 'QuantumCache'):
                return quantum_entangled_database.QuantumCache()
            logger.warning("Falling back to in-memory cache")
            return {}
        except ImportError:
            return {}

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of given text."""
        if not text.strip():
            return {"label": "NEUTRAL", "score": 0.0}
        
        result = self.sentiment_pipeline(text)
        return {
            "label": result[0]['label'].upper(),
            "score": result[0]['score']
        }

    async def entangle_news_stream(self) -> Dict[str, Any]:
        """Orchestrate quantum news entanglement process."""
        try:
            raw_data = await self._harvest_multiverse_news()
            if not raw_data:
                raise ValueError("News event horizon collapsed - no data received")
            processed = await self._quantum_processing(raw_data)
            return self._format_for_chatbot(processed)
        except Exception as e:
            logger.error(f"Quantum flux disruption: {str(e)}")
            return {"error": "Failed to stabilize news continuum"}

    async def _harvest_multiverse_news(self) -> List[Dict]:
        """Harvest news from multiple sources."""
        futures = [
            self._fetch_newsapi(),
            self._fetch_rss("reuters_gold_rss"),
            self._fetch_rss("bloomberg_metal_rss"),
            self._fetch_rss("kitco_news"),
            self._fetch_rss("goldprice_rss"),
            self._fetch_rss("tradingview_analysis"),
            self._scrape_dynamic_content("https://www.bloomberg.com/markets/commodities/gold")
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)
        return self._cleanse_reality_streams(results)

    def _cleanse_reality_streams(self, results: List) -> List[Dict]:
        """Remove temporal anomalies and duplicates."""
        clean_data = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Reality stream anomaly: {str(result)}")
                continue
            clean_data.extend(result)
        return self._remove_temporal_duplicates(clean_data)

    def _remove_temporal_duplicates(self, articles: List[Dict]) -> List[Dict]:
        """Use quantum entanglement to eliminate duplicates."""
        embeddings = self.embedder.encode([a['content'] for a in articles])
        clusters = quantum_entangled_database.quantum_cluster(embeddings)
        return [articles[i] for i in clusters]

    async def _quantum_processing(self, raw_data: List[Dict]) -> Dict:
        """Process through multiple reality layers."""
        verified = []
        for article in raw_data:
            credibility = await self._verify_article_with_quantum_entanglement(article)
            if credibility > self.temporal_convergence_threshold:
                sentiment = self.sentiment_analyzer.analyze_sentiment(article['content'])
                verified.append({
                    'title': article['title'],
                    'source': article['source'],
                    'sentiment': sentiment,
                    'credibility': credibility
                })
        return {'temporal_articles': verified}

    async def _verify_article_with_quantum_entanglement(self, article: Dict) -> float:
        """Verify news using quantum algorithms and multiple sources."""
        quantum_score = quantum_entangled_database.quantum_verify(article['content'])
        return quantum_score

    async def _fetch_newsapi(self) -> List[Dict]:
        """Fetch news from the NewsAPI."""
        url = QUANTUM_SOURCES['newsapi']['url']
        params = QUANTUM_SOURCES['newsapi']['params']
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            if response.status_code == 200:
                return [{"title": article['title'], "content": article['description'], "source": article['source']['name']} 
                        for article in response.json()['articles']]
            return []

    async def _fetch_rss(self, source: str) -> List[Dict]:
        """Fetch news from RSS feed sources."""
        # Placeholder for RSS feed fetching logic
        pass  # RSS implementation goes here...

    async def _scrape_dynamic_content(self, url: str) -> List[Dict]:
        """Scrape dynamic content with Selenium."""
        options = Options()
        options.headless = True
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'article-class')))
        articles = driver.find_elements(By.CLASS_NAME, 'article-class')
        return [{"title": a.text, "content": a.text, "source": url} for a in articles]

    def _format_for_chatbot(self, processed_data: Dict) -> Dict:
        """Format processed data for chatbot consumption."""
        return {
            "articles": processed_data.get("temporal_articles", []),
            "status": "success"
        }