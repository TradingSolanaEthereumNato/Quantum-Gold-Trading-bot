import yfinance as yf
from pycoingecko import CoinGeckoAPI
from polygon.rest import RESTClient
import pandas as pd
import logging
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from threading import Thread
import hashlib
import pytz
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
import ib_insync  # Interactive Brokers API integration
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Load environment variables
load_dotenv()

class MarketDataHandler:
    def __init__(self, symbols: List[str], start_date: str, risk_tolerance: float = 0.5):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = datetime.today().strftime('%Y-%m-%d')  # Always set end date to today's date
        self.data: Optional[Dict[str, pd.DataFrame]] = {}
        self.risk_tolerance = risk_tolerance

    def fetch_data(self, source: str = "yfinance") -> None:
        try:
            def fetch_from_source(symbol, source):
                if source == "yfinance":
                    return self.fetch_from_yfinance(symbol)
                elif source == "coingecko":
                    return self.fetch_from_coingecko(symbol)
                elif source == "polygon":
                    return self.fetch_from_polygon(symbol)
                elif source == "selenium":
                    return self.fetch_from_selenium(symbol)
                elif source == "ibkr":
                    return self.fetch_from_ibkr(symbol)
                else:
                    logger.error(f"Unsupported data source: {source}")
                    return None

            threads = []
            for symbol in self.symbols:
                t = Thread(target=fetch_from_source, args=(symbol, source))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            self.data = None
    def get_current_prices(self) -> dict:
        """Get formatted current prices for all symbols"""
        prices = {}
        for symbol, data in self.data.items():
            if not data.empty:
                prices[symbol] = {
                    'close': data['close'].iloc[-1],
                    'timestamp': data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                }
        return prices

    def get_analysis_summary(self) -> str:
        """Get formatted analysis summary for AI context"""
        analysis = self.analyze_market_data()
        summary = []
        for symbol, data in analysis.items():
            summary.append(f"{symbol}: {data['sentiment']} sentiment, "
                          f"Sharpe Ratio: {data['sharpe_ratio']:.2f}, "
                          f"Recommendation: {data['decision']}")
        return "\n".join(summary)


    def fetch_from_yfinance(self, symbol: str) -> None:
        try:
            logger.info(f"Fetching data from Yahoo Finance for {symbol}...")
            df = yf.download(symbol, start=self.start_date, end=self.end_date)
            df.reset_index(inplace=True)
            self.data[symbol] = df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from Yahoo Finance: {e}")

    def fetch_from_coingecko(self, symbol: str) -> None:
        try:
            logger.info(f"Fetching data from CoinGecko for {symbol}...")
            cg = CoinGeckoAPI()
            data = cg.get_coin_market_chart_range_by_id(
                id=symbol.lower(),
                vs_currency="usd",
                from_timestamp=int(pd.to_datetime(self.start_date).timestamp()),
                to_timestamp=int(pd.to_datetime(self.end_date).timestamp())
            )
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            self.data[symbol] = df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from CoinGecko: {e}")

    def fetch_from_polygon(self, symbol: str) -> None:
        try:
            logger.info(f"Fetching data from Polygon.io for {symbol}...")
            POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
            if not POLYGON_API_KEY:
                raise ValueError("Polygon API key is missing.")
            client = RESTClient(POLYGON_API_KEY)

            if symbol == "GOLD":
                symbol = "XAU-USD"
            historical_data = client.get_aggs(symbol, 1, "day", self.start_date, self.end_date)
            if historical_data:
                df = pd.DataFrame(historical_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                self.data[symbol] = df
            else:
                raise ValueError(f"No data found for symbol: {symbol} in Polygon.")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from Polygon: {e}")

    def fetch_from_selenium(self, symbol: str) -> None:
        try:
            logger.info(f"Scraping live data for {symbol} using Selenium...")
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
            driver.get(f"https://www.example.com/marketdata/{symbol}")  # Replace with actual market data URL
            price = driver.find_element(By.XPATH, "//*[@id='price']").text  # Update with actual XPath
            df = pd.DataFrame({"timestamp": [datetime.now()], "close": [price]})
            df.set_index('timestamp', inplace=True)
            self.data[symbol] = df
            driver.quit()
        except Exception as e:
            logger.error(f"Error scraping data for {symbol} using Selenium: {e}")

    def fetch_from_ibkr(self, symbol: str) -> None:
        try:
            logger.info(f"Fetching data from IBKR for {symbol}...")
            IBKR_TWS_HOST = os.getenv('IBKR_TWS_HOST')
            IBKR_TWS_PORT = os.getenv('IBKR_TWS_PORT')
            ibkr_client = ib_insync.IB()
            ibkr_client.connect(IBKR_TWS_HOST, IBKR_TWS_PORT, clientId=1)

            contract = ib_insync.Stock(symbol, 'SMART', 'USD')  # Replace with appropriate contract type for commodities
            market_data = ibkr_client.reqMktData(contract)
            df = pd.DataFrame({
                "timestamp": [datetime.now()],
                "close": [market_data.last]
            })
            df.set_index('timestamp', inplace=True)
            self.data[symbol] = df
            ibkr_client.disconnect()
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from IBKR: {e}")

    def analyze_market_data(self) -> Dict[str, Any]:
        if self.data:
            analysis_results = {}
            for symbol, df in self.data.items():
                try:
                    scaler = StandardScaler()
                    df['close'] = scaler.fit_transform(df[['close']])
                    df['ema'] = df['close'].ewm(span=30, adjust=False).mean()
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(df[['close']])
                    df['pca1'] = pca_result[:, 0]
                    df['pca2'] = pca_result[:, 1]

                    model = ARIMA(df['close'], order=(5, 1, 0))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=5)
                    df['forecast'] = np.concatenate([df['close'], forecast])

                    sentiment_analyzer = SentimentIntensityAnalyzer()
                    sentiment_scores = [sentiment_analyzer.polarity_scores("Sample news headline")['compound']]
                    avg_sentiment = np.mean(sentiment_scores)
                    sentiment_label = 'bullish' if avg_sentiment > 0 else 'bearish'

                    returns = df['close'].pct_change()
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                    decision = 'BUY' if avg_sentiment > 0.5 else 'SELL'

                    avg_price = np.mean(df['close'])
                    median_price = np.median(df['close'])

                    market_trends = {
                        'sentiment': sentiment_label,
                        'sharpe_ratio': sharpe_ratio,
                        'decision': decision,
                        'forecasted_close': forecast[-1],
                        'avg_price': avg_price,
                        'median_price': median_price,
                    }
                    analysis_results[symbol] = market_trends
                    logger.info(f"📈 Market Analysis for {symbol}: {market_trends}")
                except Exception as e:
                    logger.error(f"Error analyzing data for {symbol}: {e}")
            return analysis_results
        else:
            logger.warning(f"No market data available.")
            return {}

    def process_user_query(self, query: str) -> str:
        response = client.completions.create(model="gpt-3.5-turbo",
        prompt=query,
        max_tokens=150)
        return response.choices[0].text.strip()
# quantum_predictor.py
try:
    from qiskit import QuantumCircuit, execute, Aer
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

class QuantumPredictor:
    def __init__(self):
        if not HAS_QISKIT:
            raise RuntimeError("Qiskit not installed. Quantum features disabled.")

    def run_predictions(self):
        if not HAS_QISKIT:
            return None  # Return default value
        # Existing quantum logic

if __name__ == "__main__":
    symbols = ['AAPL', 'BTC', 'XAU-USD']
    start_date = '2000-01-01'
    market_data_handler = MarketDataHandler(symbols, start_date)
    market_data_handler.fetch_data(source='yfinance')  # Choose from 'yfinance', 'coingecko', 'polygon', 'selenium', or 'ibkr'

    analysis_results = market_data_handler.analyze_market_data()

    query = "What is the predicted trend for gold in 2025?"
    chatbot_response = market_data_handler.process_user_query(query)
    logger.info(f"ChatGPT Response: {chatbot_response}")


