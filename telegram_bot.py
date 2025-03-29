import os
import logging
import threading
import numpy as np
import tensorflow as tf
from openai import OpenAI
from dotenv import load_dotenv
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters,
    CallbackQueryHandler, ContextTypes, ConversationHandler
)
import yfinance as yf
import xgboost as xgb
from lightgbm import LGBMRegressor
from prophet import Prophet
from sktime.transformations.panel.rocket import Rocket
from datetime import datetime
from ai_optimizer import AIOptimizer, Indicators
from quantum_predictor import QuantumPredictor
from ml_predictor import MilitaryMLModel
from security import SecurityManager
import sqlite3
import requests
import warnings
from retry import retry
from ib_insync import IB, Contract, util  # Added IBKR connection
import pickle
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
from tensorflow.keras import layers, models
from tensorflow.keras.losses import MeanSquaredError

def custom_mse(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)


###### Suppress non-critical warnings #####
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

###### --- INITIALIZATION ---
load_dotenv()
required_env_vars = ['BOT_TOKEN', 'USER_ID', 'OPENAI_API_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# For optional APIs
if not os.getenv('POLYGON_API_KEY'):
    logger.warning("Polygon.io API key not found - limited market data available")
if not os.getenv('COINGECKO_API_KEY'):
    logger.warning("CoinGecko API key not found - crypto data may be limited")

# --- LOGGING ---
logger = logging.getLogger('ModelTraining')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


#### Config ####
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure model directory exists

MODEL_PATHS = {
    'tft': os.path.join(MODEL_DIR, "tft.h5"),
    'xgboost': os.path.join(MODEL_DIR, "xgb.model"),
    'prophet': os.path.join(MODEL_DIR, "prophet.bin")
}



##### --- CONSTANTS ---

CONFIG = {
    'SYMBOLS': ['GC=F', 'XAUUSD=X', 'BTC-USD', 'ETH-USD'],
    'MAX_TRADE_AMOUNT': 1000000,
    'RISK_THRESHOLD': 0.85,
    'DATA_REFRESH_INTERVAL': 300,
    'MAX_POSITION_SIZE': 0.1,
    'IBKR_HOST': '127.0.0.1',
    'IBKR_PORT': 7497,
    'COINGECKO_SYMBOL_MAP': {
        'BTC-USD': 'bitcoin',
        'ETH-USD': 'ethereum',
        'GC=F': 'gold',
        'XAUUSD=X': 'gold'
    }
}


CONFIG = {
    'SYMBOLS': ['GC=F', 'XAUUSD=X', 'BTC-USD', 'ETH-USD'],
    'MAX_TRADE_AMOUNT': 1000000,
    'RISK_THRESHOLD': 0.85,
    'DATA_REFRESH_INTERVAL': 300,
    'MAX_POSITION_SIZE': 0.1
}

STATE_MENU = 0
STATE_STRATEGY = 1
STATE_RISK = 2
STATE_TRADE = 3
STATE_AI = 4
STATE_POSITION_TYPE = 5

VALID_ASSETS = {'BTC', 'ETH', 'XAU', 'AAPL', 'GOOGL'}
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
USER_ID = os.getenv("TELEGRAM_USER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")


def fetch_market_data(symbol: str, period: str = "5y", interval: str = "1d"):
    """Fetch historical market data using yfinance."""
    try:
        logger.info(f"Fetching market data for {symbol}...")
        data = yf.download(symbol, period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch market data: {str(e)}")
        return None


def fetch_market_data(symbol: str, period: str = "5y", interval: str = "1d"):
    """Fetch historical market data using yfinance with fallback symbols."""
    fallback_symbols = {
        'GC=F': ['GC=F', 'GC%3DF', 'GC'],  # CME Gold futures with URL-encoded alternative
        'XAUUSD=X': ['XAUUSD=X', 'XAUUSD', 'XAU-USD'],
        'BTC-USD': ['BTC-USD', 'BTCUSD=X', 'BTCUSDT'],
        'ETH-USD': ['ETH-USD', 'ETHUSD=X', 'ETHUSDT']
    }
    
    logger.debug(f"Attempting to fetch data for symbol: {symbol}")
    logger.debug(f"Fallback symbols: {fallback_symbols.get(symbol, [symbol])}")
    
    for attempt in range(3):  # Retry up to 3 times
        for symbol_variant in fallback_symbols.get(symbol, [symbol]):
            try:
                logger.info(f"Fetching market data for {symbol_variant}...")
                data = yf.download(symbol_variant, period=period, interval=interval)
                if not data.empty:
                    logger.debug(f"Successfully fetched data for {symbol_variant}")
                    return data
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol_variant}: {str(e)}")
                continue
    
    raise ValueError(f"All data sources failed for {symbol}")


### Model Training Functions

def _train_tft():
    """Train Temporal Fusion Transformer (TFT) using real market data."""
    logger.info("🚀 Training Temporal Fusion Transformer (TFT) with real market data...")

    # Fetch real market data
    symbol = "GC=F"  # Gold futures
    data = fetch_market_data(symbol)
    if data is None:
        logger.error("Failed to fetch market data for TFT training.")
        return

    # Prepare data for TFT
    close_prices = data['Close'].values
    time_steps = 30  # Use 30 days of historical data for each prediction

    X, y = [], []
    for i in range(len(close_prices) - time_steps):
        X.append(close_prices[i:i + time_steps])
        y.append(close_prices[i + time_steps])

    X = np.array(X).reshape(-1, time_steps, 1)
    y = np.array(y)

    # Build TFT model
    inputs = layers.Input(shape=(time_steps, 1))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='linear')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=custom_mse)

    # Train the model
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
    model.save(MODEL_PATHS['tft'])

    logger.info(f"✅ TFT model saved: {MODEL_PATHS['tft']}")


def _train_xgboost():
    """Train XGBoost model using real market data."""
    logger.info("🚀 Training XGBoost model with real market data...")

    # Fetch real market data
    symbol = "GC=F"  # Gold futures
    data = fetch_market_data(symbol)
    if data is None:
        logger.error("Failed to fetch market data for XGBoost training.")
        return

    # Feature engineering
    data['Returns'] = data['Close'].pct_change()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['Volatility'] = data['Returns'].rolling(window=10).std()
    data.dropna(inplace=True)

    # Prepare features and target
    features = ['Close', 'SMA_10', 'SMA_50', 'Volatility']
    X = data[features]
    y = data['Close'].shift(-1).dropna()  # Predict next day's close price
    X = X.iloc[:-1]  # Align X and y

    # Train XGBoost model
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, random_state=42)
    model.fit(X, y)

    # Save the model
    model.save_model(MODEL_PATHS['xgboost'])
    logger.info(f"✅ XGBoost model saved: {MODEL_PATHS['xgboost']}")


def _train_prophet():
    """Train Prophet model using real market data."""
    logger.info("🚀 Training Prophet model with real market data...")

    # Fetch real market data
    symbol = "GC=F"  # Gold futures
    data = fetch_market_data(symbol)
    if data is None:
        logger.error("Failed to fetch market data for Prophet training.")
        return

    # Prepare data for Prophet
    df = data[['Close']].reset_index()
    df.columns = ['ds', 'y']  # Prophet requires 'ds' (date) and 'y' (target)

    # Train Prophet model
    model = Prophet()
    model.fit(df)

    # Save the model
    with open(MODEL_PATHS['prophet'], "wb") as f:
        pickle.dump(model, f)

    logger.info(f"✅ Prophet model saved: {MODEL_PATHS['prophet']}")


### Model Loading Function

def load_models():
    """Load trained models or train them if missing."""
    logger.info("🔄 Loading trained models...")

    # Train models if missing
    if not os.path.exists(MODEL_PATHS['tft']):
        _train_tft()
    if not os.path.exists(MODEL_PATHS['xgboost']):
        _train_xgboost()
    if not os.path.exists(MODEL_PATHS['prophet']):
        _train_prophet()

    # Load models
    custom_objects = {"custom_mse": custom_mse}  # Pass the custom loss function
    tft_model = tf.keras.models.load_model(MODEL_PATHS['tft'], custom_objects=custom_objects)
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(MODEL_PATHS['xgboost'])
    with open(MODEL_PATHS['prophet'], "rb") as f:
        prophet_model = pickle.load(f)

    logger.info("✅ All models loaded successfully.")
    return tft_model, xgb_model, prophet_model


# --- SECURITY CHECK ---
if not all([BOT_TOKEN, USER_ID, OPENAI_API_KEY]):
    raise EnvironmentError("Missing required environment variables")

# --- DATABASE SETUP ---
conn = sqlite3.connect('trading_bot.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS market_data (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    price REAL,
    volume REAL,
    timestamp DATETIME,
    source TEXT
)""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    prediction REAL,
    confidence REAL,
    timestamp DATETIME
)""")

conn.commit()

# --- ENHANCED LOGGING ---
class InstitutionalFormatter(logging.Formatter):
    def format(self, record):
        record.timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S.%f')[:-3]
        return super().format(record)

logger = logging.getLogger('GOLD_TRADER')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(InstitutionalFormatter(
    '%(timestamp)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(handler)

# --- SECURITY ENCLAVE ---
class SecurityEnclave:
    @staticmethod
    def sanitize_input(data: np.ndarray) -> np.ndarray:
        """Sanitize input data to handle NaNs and infinities."""
        return np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)

    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate trading symbol."""
        return symbol in CONFIG['SYMBOLS']

# --- INDICATORS CLASS ---
class Indicators:
    @staticmethod
    def ema(data: np.ndarray, period: int) -> float:
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return np.convolve(data[:, 3], weights, mode='valid')[-1]

    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> float:
        delta = np.diff(data[:, 3])
        gain = delta.copy()
        gain[gain < 0] = 0.0
        loss = -delta.copy()
        loss[loss < 0] = 0.0
        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        close = data[:, 3]
        fast_ema = np.convolve(close, np.ones(fast)/fast, mode='valid')[-1]
        slow_ema = np.convolve(close, np.ones(slow)/slow, mode='valid')[-1]
        macd_line = fast_ema - slow_ema
        signal_line = np.convolve(close[-signal:], np.ones(signal)/signal)[-1]
        return macd_line, signal_line

# --- RISK MANAGER CLASS ---
class RiskManager:
    def __init__(self, risk_tolerance=0.5, max_risk=20.0, stop_loss_percentage=2.0, take_profit_percentage=5.0):
        self.risk_tolerance = risk_tolerance
        self.max_risk = max_risk
        self.stop_loss_percentage = stop_loss_percentage
        self.take_profit_percentage = take_profit_percentage
        self.logger = logging.getLogger('RiskManager')
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        self.logger.addHandler(handler)

    def calculate_risk(self, account_balance, trade_amount):
        """Calculate risk with validation."""
        if account_balance <= 0 or trade_amount <= 0:
            raise ValueError("Invalid account balance or trade amount")
        risk_amount = account_balance * (trade_amount / 100) * (self.stop_loss_percentage / 100)
        self.logger.debug(f"Risk calculated: {risk_amount} for trade amount {trade_amount}%")
        return risk_amount

    def calculate_position_size(self, prediction_confidence: float) -> float:
        """Dynamic position sizing using volatility-adjusted Kelly"""
        volatility = self._calculate_volatility()
        kelly_fraction = prediction_confidence / volatility
        return min(
            CONFIG['MAX_POSITION_SIZE'],
            kelly_fraction * self.risk_tolerance
        )

    def validate_trade(self, account_balance, trade_amount):
        """Validate trade against risk limits."""
        risk_amount = self.calculate_risk(account_balance, trade_amount)
        if risk_amount > account_balance * (self.max_risk / 100):
            self.logger.warning(f"Trade exceeds max risk: {trade_amount}%")
            return False
        return True

    def adjust_trade(self, account_balance, trade_amount):
        if not self.validate_trade(account_balance, trade_amount):
            adjusted_trade = (self.max_risk / self.stop_loss_percentage) * account_balance
            self.logger.info(f"Adjusting trade amount to: {adjusted_trade} based on max risk tolerance.")
            return adjusted_trade
        return trade_amount

    def apply_stop_loss(self, entry_price):
        stop_loss_price = entry_price * (1 - self.stop_loss_percentage / 100)
        self.logger.debug(f"Stop loss price set at: {stop_loss_price} for entry price {entry_price}.")
        return stop_loss_price

    def apply_take_profit(self, entry_price):
        take_profit_price = entry_price * (1 + self.take_profit_percentage / 100)
        self.logger.debug(f"Take profit price set at: {take_profit_price} for entry price {entry_price}.")
        return take_profit_price

# --- INITIALIZE COMPONENTS ---
risk_manager = RiskManager()
indicators = Indicators()
quantum_predictor = QuantumPredictor()
ml_model = MilitaryMLModel()
security_manager = SecurityManager()
ai_predictor = AIOptimizer(risk_manager=risk_manager, indicators=indicators)

# --- LOGGING SETUP ---
class SecureFormatter(logging.Formatter):
    def format(self, record):
        record.msg = f"[SECURE] {record.msg}"
        return super().format(record)

logger = logging.getLogger('GOLD_TRADER')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(SecureFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


# --- AI OPTIMIZER ---
class AIOptimizer:
    def __init__(self, risk_manager, indicators):
        self.lock = threading.RLock()
        self.security_layer = SecurityEnclave()
        self.risk_manager = risk_manager
        self.indicators = indicators
        self.data_engine = self.initialize_data_engine()
        self.models = self.initialize_ai_models()
        self._system_checks()
        self.models = self._load_models()

    def _system_checks(self):
        """Check system capabilities and log warnings."""
        if not tf.config.list_physical_devices('GPU'):
            logger.warning("GPU acceleration not available")
        if not all(key in os.environ for key in ['OPENAI_API_KEY', 'POLYGON_API_KEY']):
            logger.warning("Missing required API keys in environment variables")

    def initialize_ai_models(self):
        """Initialize all AI models with proper error handling."""
        try:
            return {
                "temporal_fusion": TemporalFusionTransformer(60, 42),
                "xgboost": xgb.XGBRegressor(n_estimators=1000),
                "prophet": Prophet(),
                "rocket_transform": Rocket(num_kernels=10000)
            }
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {str(e)}")
            return {}

    def initialize_data_engine(self):
        return {'market': self._init_market_data()}

    def _init_market_data(self):
        """Fetch initial market data with error handling."""
        with self.lock:
            market_data = {}
            for symbol in CONFIG['SYMBOLS']:
                try:
                    market_data[symbol] = yf.Ticker(symbol)
                except Exception as e:
                    logger.error(f"Failed to initialize market data for {symbol}: {str(e)}")
            return market_data


    def _load_models(self):
        """Load models using the model manager."""
        try:
            tft_model, xgb_model, prophet_model = load_models()
            return {
                'tft': tft_model,
                'xgboost': xgb_model,
                'prophet': prophet_model
            }
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            return self.initialize_fallback_models()

    def initialize_fallback_models(self):
        """Initialize fallback models if loading fails."""
        return {
            'tft': TemporalFusionTransformer(60, 42),
            'xgboost': xgb.XGBRegressor(n_estimators=100),
            'prophet': Prophet()
        }

    def _check_and_train_models(self):
        """Train models if they're missing"""
        if not os.path.exists(MODEL_PATHS['tft']):
            self._train_tft()
        if not os.path.exists(MODEL_PATHS['xgboost']):
            self._train_xgboost()
        if not os.path.exists(MODEL_PATHS['prophet']):
            self._train_prophet()


    def initialize_with_training(self):
        """Initialize models with basic training if no pretrained available"""
        for name, model in self.models.items():
            if isinstance(model, xgb.XGBRegressor):
                # Train with dummy data
                X = np.random.rand(100, 10)
                y = np.random.rand(100)
                model.fit(X, y)
            elif isinstance(model, Prophet):
                df = pd.DataFrame({
                    'ds': pd.date_range(end=datetime.today(), periods=365),
                    'y': np.random.rand(365)
                })
                model.fit(df)


    def _load_tft_model(self, path):
        return tf.keras.models.load_model(path)

    def _load_xgboost_model(self, path):
        return xgb.Booster(model_file=path)

    def _load_prophet_model(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def ensemble_prediction(self, processed_data: np.ndarray) -> float:
        """Generate ensemble predictions with validation."""
        if processed_data.size == 0:
            logger.error("No processed data available for prediction")
            return 0.0

        predictions = []
        for name, model in self.models.items():
            try:
                if isinstance(model, tf.keras.Model):
                    pred = model.predict(processed_data[-1].reshape(1, 60, -1))[0][0]
                else:
                    pred = model.predict(processed_data[-1].reshape(1, -1))[0]
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Prediction failed for model {name}: {str(e)}")
        return np.mean(predictions) if predictions else 0.0

    def trade(self):
        while True:
            market_data = self.fetch_market_data()
            processed_data = self.temporal_fold_processing(market_data)
            prediction = self.ensemble_prediction(processed_data)
            self.risk_manager.execute_trade(prediction, 
                self.risk_manager.calculate_position_size(prediction))
            time.sleep(CONFIG['DATA_REFRESH_INTERVAL'])

    def fetch_market_data(self) -> np.ndarray:
        all_data = []
        for symbol in CONFIG['SYMBOLS']:
            history = yf.Ticker(symbol).history(period="1d", interval="5m")
            all_data.append(history[['Open', 'High', 'Low', 'Close', 'Volume']].values)
        return np.concatenate(all_data, axis=0)

    def process_query(self, query: str) -> str:
        """Process user queries using OpenAI."""
        client = OpenAI(api_key=OPENAI_API_KEY)
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a seasoned financial trading expert with deep expertise in gold markets, specializing in high-frequency trading, algorithmic strategies, and quantitative analysis. Your knowledge spans macroeconomic trends, geopolitical influences, central bank policies, and market sentiment analysis that impact gold prices. You excel at interpreting technical indicators, AI-powered trading models, and quantum risk management techniques. Your role is to provide real-time insights, predictive analytics, and actionable trade recommendations tailored to professional traders, hedge funds, and institutional investors. Ensure your analysis is data-driven, leveraging advanced statistical models, historical patterns, and emerging market dynamics to optimize profitability and risk-adjusted returns in gold trading."},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"AI Error: {str(e)}")
            return "Unable to process request at this time."

    def temporal_fold_processing(self, data: np.ndarray) -> np.ndarray:
        processed = []
        for i in range(len(data)-60):
            window = data[i:i+60]
            features = [
                self.indicators.ema(window, 20),
                self.indicators.rsi(window),
                self.indicators.macd(window)
            ]
            processed.append(np.concatenate(features))
        return np.array(processed)

class TemporalFusionTransformer:
    def __init__(self, time_steps: int, features: int):
        self.time_steps = time_steps
        self.features = features
        self.model = self.build_tft(time_steps, features)

    def build_tft(self, time_steps: int, features: int) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(time_steps, features))
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True))(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        attn = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
        x = tf.keras.layers.Add()([x, attn])
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        return model

    def save(self, path: str):
        """Save the TFT model to the specified path."""
        self.model.save(path)

    def load(self, path: str):
        """Load the TFT model from the specified path."""
        self.model = tf.keras.models.load_model(path)


# --- MARKET DATA HANDLER ---
class MarketDataHandler:
    def __init__(self):
        self.sources = {
            'yfinance': self._fetch_yfinance,
            'polygon': self._fetch_polygon,
            'coingecko': self._fetch_coingecko
        }
        self.data = {}
        self.logger = logging.getLogger('MarketDataHandler')
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(SecureFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def __init__(self):
        self.fallback_symbols = {
            'XAUUSD=X': 'GC=F',
            'GC=F': 'XAUUSD=X',
            'BTC-USD': 'BTC-USD',
            'ETH-USD': 'ETH-USD'
        }
        
    @retry(tries=3, delay=2, backoff=2)
    def fetch_symbol_data(self, symbol: str):
        """Enhanced data fetcher with fallback symbols"""
        try:
            # Try primary symbol
            data = self._fetch_yfinance(symbol)
            if data and self._validate_market_response(data):
                return data
                
            # Fallback to alternative symbol
            fallback = self.fallback_symbols.get(symbol)
            if fallback:
                logger.warning(f"Using fallback symbol {fallback} for {symbol}")
                data = self._fetch_yfinance(fallback)
                if data and self._validate_market_response(data):
                    return data
                    
            raise ValueError(f"All data sources failed for {symbol}")
            
        except Exception as e:
            logger.error(f"Data fetch failed for {symbol}: {str(e)}")
            return None
                
        raise ValueError(f"All data sources failed for {symbol}")

    def fetch_market_data(self) -> np.ndarray:
        """Fetch market data for all symbols with proper fallbacks"""
        all_data = []
        for symbol in CONFIG['SYMBOLS']:
            try:
                data = self.fetch_symbol_data(symbol)
                all_data.append([
                    data['price'],
                    data.get('high', data['price']),
                    data.get('low', data['price']),
                    data['price'],  # Close price
                    data['volume']
                ])
            except Exception as e:
                self.logger.error(f"Final data failure for {symbol}: {str(e)}")
        
        return np.array(all_data) if all_data else np.array([])

    def __init__(self):
        self.symbol_map = {
            'GC=F': ['GC=F', 'GC%3DF', 'GC'],  # CME Gold futures with URL-encoded alternative
            'XAUUSD=X': ['XAUUSD=X', 'XAUUSD', 'XAU-USD'],
            'BTC-USD': ['BTC-USD', 'BTCUSD=X', 'BTCUSDT'],
            'ETH-USD': ['ETH-USD', 'ETHUSD=X', 'ETHUSDT']
        }
        self.data_sources = ['yfinance', 'polygon', 'coingecko']
        self.logger = logging.getLogger('MarketDataHandler')

    @retry(tries=3, delay=2, backoff=2, logger=logger)
    def fetch_data(self, symbol: str) -> dict:
        """Robust multi-source data fetching"""
        original_symbol = symbol
        for attempt in range(3):
            for symbol_variant in self.symbol_map.get(original_symbol, [original_symbol]):
                try:
                    # Try Yahoo Finance first
                    ticker = yf.Ticker(symbol_variant)
                    hist = ticker.history(period="5d", interval="1h")
                    
                    if not hist.empty and 'Close' in hist.columns:
                        return {
                            'price': hist['Close'].iloc[-1],
                            'high': hist['High'].iloc[-1],
                            'low': hist['Low'].iloc[-1],
                            'volume': hist['Volume'].iloc[-1],
                            'timestamp': datetime.now()
                        }
                    
                    # Fallback to Polygon if available
                    if POLYGON_API_KEY:
                        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol_variant}/prev"
                        response = requests.get(url, params={'apiKey': POLYGON_API_KEY}).json()
                        if response.get('results'):
                            return {
                                'price': response['results'][0]['c'],
                                'high': response['results'][0]['h'],
                                'low': response['results'][0]['l'],
                                'volume': response['results'][0]['v'],
                                'timestamp': datetime.now()
                            }
                    
                    # Fallback to CoinGecko for crypto
                    if 'USD' in symbol_variant and COINGECKO_API_KEY:
                        cg_symbol = CONFIG['COINGECKO_SYMBOL_MAP'].get(original_symbol)
                        if cg_symbol:
                            url = f"https://api.coingecko.com/api/v3/simple/price"
                            params = {'ids': cg_symbol, 'vs_currencies': 'usd'}
                            response = requests.get(url, params=params).json()
                            return {
                                'price': response[cg_symbol]['usd'],
                                'timestamp': datetime.now()
                            }
                            
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt+1} failed for {symbol_variant}: {str(e)}")
                    continue
        
        raise ValueError(f"All data sources failed for {original_symbol}")

    def get_market_report(self):
        """Generate comprehensive market report"""
        report = []
        for symbol in CONFIG['SYMBOLS']:
            try:
                data = self.fetch_data(symbol)
                report.append(
                    f"{symbol}:\n"
                    f"  Price: {data['price']:.2f}\n"
                    f"  High: {data.get('high', data['price']):.2f}\n"
                    f"  Low: {data.get('low', data['price']):.2f}\n"
                    f"  Volume: {data.get('volume', 'N/A')}"
                )
            except Exception as e:
                report.append(f"{symbol}: Data unavailable ({str(e)})")
        return "\n\n".join(report)

    def _validate_market_response(self, data: dict) -> bool:
        """Enhanced validation with price sanity checks"""
        return all([
            data['price'] > 0,
            data['volume'] >= 0,
            datetime.now() - data['timestamp'] < timedelta(minutes=15),
            data['price'] < 100000  # Sanity check for gold prices
        ])

    @retry(tries=3, delay=2, backoff=2)
    def _fetch_yfinance(self, symbol: str):
        """Enhanced Yahoo Finance data fetcher with validation"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1h")
            
            if hist.empty:
                raise ValueError(f"No data available for {symbol}")
                
            if 'Close' not in hist.columns or 'Volume' not in hist.columns:
                raise ValueError(f"Missing required columns in data for {symbol}")

            return {
                'price': hist['Close'].iloc[-1],
                'volume': hist['Volume'].iloc[-1],
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"YFinance error for {symbol}: {str(e)}")
            raise


    @retry(tries=3, delay=2)
    def _fetch_polygon(self, symbol: str):
        """Fetch data from Polygon.io with retry logic."""
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
            params = {'apiKey': POLYGON_API_KEY}
            response = requests.get(url, params=params).json()
            if 'results' not in response or not response['results']:
                raise ValueError(f"No data available for {symbol}")
            return {
                'price': response['results'][0]['c'],
                'volume': response['results'][0]['v'],
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Polygon error for {symbol}: {str(e)}")
            raise

    def fetch_data(self, source='yfinance'):
        """Enhanced data fetcher with multiple fallbacks"""
        try:
            # Try primary symbol first
            spot_price_data = yf.Ticker("XAUUSD=X").history(period="1d")
            fix_price_data = yf.Ticker("GC=F").history(period="1d")

            # Fallback to alternative symbols if needed
            if spot_price_data.empty:
                spot_price_data = yf.Ticker("GC=F").history(period="1d")
            if fix_price_data.empty:
                fix_price_data = yf.Ticker("XAUUSD=X").history(period="1d")

            # Validate data after fallback
            if spot_price_data.empty or fix_price_data.empty:
                raise ValueError("No data available after fallback attempts")

            if 'Close' not in spot_price_data.columns or 'Close' not in fix_price_data.columns:
                raise ValueError("Missing 'Close' column in market data")

            spot_price = spot_price_data['Close'].iloc[-1]
            fix_price = fix_price_data['Close'].iloc[-1]

            self.data['spot_price'] = spot_price
            self.data['fix_price'] = fix_price

            self.logger.info(f"Spot Price (XAU to USD): ${spot_price:.2f}")
            self.logger.info(f"Fix Price (XAU to USD): ${fix_price:.2f}")

            return (
                f"📊 Current Gold Prices:\n"
                f"🔸 Spot Price (XAU to USD): ${spot_price:.2f}\n"
                f"🔸 Fix Price (XAU to USD): ${fix_price:.2f}"
            )
        except Exception as e:
            self.logger.error(f"Final data fetch error: {e}")
            return "⚠️ Market data temporarily unavailable. Please try again later."

    @retry(tries=3, delay=2)
    def _fetch_coingecko(self, symbol: str):
        """Fetch data from CoinGecko with retry logic."""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {'ids': symbol.lower(), 'vs_currencies': 'usd', 'include_market_cap': 'true'}
            response = requests.get(url, params=params).json()
            if symbol.lower() not in response:
                raise ValueError(f"No data available for {symbol}")
            return {
                'price': response[symbol.lower()]['usd'],
                'volume': response[symbol.lower()]['usd_market_cap'],
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"CoinGecko error for {symbol}: {str(e)}")
            raise

    def _validate_market_response(self, data: dict) -> bool:
        """Ensure data meets quality standards"""
        return all([
            data['price'] > 0,
            data['volume'] >= 0,
            datetime.now() - data['timestamp'] < timedelta(minutes=5)
        ])

    def fetch_market_data(self) -> np.ndarray:
        """Fetch and validate market data for all symbols."""
        all_data = []
        for symbol in CONFIG['SYMBOLS']:
            try:
                data = self._fetch_yfinance(symbol)
                if data:
                    all_data.append(data)
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
        return np.array(all_data) if all_data else np.array([])

    def fetch_ibkr_data(self, symbol: str):
        """Fetch data for a given symbol from IBKR (mock implementation)."""
        try:
            # Mock implementation using yfinance
            data = yf.Ticker(symbol).history(period="1d")
            if data.empty:
                raise ValueError(f"No data available for {symbol}.")
            return data['Close'].iloc[-1]
        except Exception as e:
            self.logger.error(f"Error fetching IBKR data for {symbol}: {e}")
            return None

    def fetch_fix_price(self):
        """Fetch the LBMA Gold Fix price."""
        try:
            fix_price_data = yf.Ticker("GC=F").history(period="1d")
            if fix_price_data.empty:
                raise ValueError("No data available for GC=F.")
            return fix_price_data['Close'].iloc[-1]
        except Exception as e:
            self.logger.error(f"Error fetching fix price: {e}")
            return None

    def fetch_coingecko_data(self):
        """Fetch cryptocurrency data from CoinGecko."""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': 'bitcoin,ethereum',
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            response = requests.get(url, params=params).json()
            return response
        except Exception as e:
            self.logger.error(f"Error fetching CoinGecko data: {e}")
            return None

    def __init__(self):
        self.sources = {
            'yfinance': self._fetch_yfinance,
            'polygon': self._fetch_polygon,
            'coingecko': self._fetch_coingecko
        }
        self.data = {}  # Initialize data dictionary
        self.logger = logging.getLogger('MarketDataHandler')  # Add logger
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(SecureFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def fetch_market_data(self) -> np.ndarray:
        """Fetch market data for all symbols."""
        all_data = []
        for symbol in CONFIG['SYMBOLS']:
            history = yf.Ticker(symbol).history(period="1d", interval="5m")
            all_data.append(history[['Open', 'High', 'Low', 'Close', 'Volume']].values)
        return np.concatenate(all_data, axis=0)


    def _fetch_yfinance(self, symbol: str):
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        return {
            'price': hist['Close'].iloc[-1],
            'volume': hist['Volume'].iloc[-1],
            'timestamp': datetime.now()
        }

    def _fetch_polygon(self, symbol: str):
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
        params = {'apiKey': POLYGON_API_KEY}
        response = requests.get(url, params=params).json()
        return {
            'price': response['results'][0]['c'],
            'volume': response['results'][0]['v'],
            'timestamp': datetime.now()
        }

    def _fetch_coingecko(self, symbol: str):
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {'ids': symbol.lower(), 'vs_currencies': 'usd', 'include_market_cap': 'true'}
        response = requests.get(url, params=params).json()
        return {
            'price': response[symbol.lower()]['usd'],
            'volume': response[symbol.lower()]['usd_market_cap'],
            'timestamp': datetime.now()
        }



    def _store_market_data(self, symbol: str, data: dict, source: str):
        cursor.execute("""
            INSERT INTO market_data (symbol, price, volume, timestamp, source)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, data['price'], data['volume'], data['timestamp'], source))
        conn.commit()

    def fetch_data(self, source='yfinance'):
        try:
            spot_price_data = yf.Ticker("XAUUSD=X").history(period="1d")
            fix_price_data = yf.Ticker("GC=F").history(period="1d")

            # Check if data is available and contains at least one row
            if spot_price_data.empty or fix_price_data.empty:
                raise ValueError("No data available for the specified symbols.")

            if 'Close' not in spot_price_data.columns or 'Close' not in fix_price_data.columns:
                raise ValueError("Missing 'Close' column in market data.")

            if len(spot_price_data['Close']) == 0 or len(fix_price_data['Close']) == 0:
                raise ValueError("Market data does not contain enough records.")

            spot_price = spot_price_data['Close'].iloc[-1]
            fix_price = fix_price_data['Close'].iloc[-1]

            self.data['spot_price'] = spot_price
            self.data['fix_price'] = fix_price

            self.logger.info(f"Spot Price (XAUUSD=X): {spot_price}")
            self.logger.info(f"Fix Price (GC=F): {fix_price}")

            return (
                f"📊 Current Gold Prices:\n"
                f"🔸 Spot Price (XAU to USD): ${spot_price:.2f}\n"
                f"🔸 Fix Price (XAU to USD): ${fix_price:.2f}"
            )
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return "Error fetching market data. Please try again later."

    def __init__(self):
        self.sources = {
            'yfinance': self._fetch_yfinance,
            'polygon': self._fetch_polygon,
            'coingecko': self._fetch_coingecko
        }
        self.data = {}  # Initialize data dictionary
        self.logger = logging.getLogger('MarketDataHandler')  # Add logger
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(SecureFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def analyze_market_data(self):
        if not self.data:
            self.logger.warning("No market data to analyze")
            return None
        
        analysis = {
            "spot_price": self.data['spot_price'],
            "fix_price": self.data['fix_price'],
            "price_difference": self.data['spot_price'] - self.data['fix_price']
        }
        
        self.logger.info(f"Market Analysis: {analysis}")
        return analysis

    def process_user_query(self, query):
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a seasoned financial trading expert with deep expertise in gold markets, specializing in high-frequency trading, algorithmic strategies, and quantitative analysis. Your knowledge spans macroeconomic trends, geopolitical influences, central bank policies, and market sentiment analysis that impact gold prices. You excel at interpreting technical indicators, AI-powered trading models, and quantum risk management techniques. Your role is to provide real-time insights, predictive analytics, and actionable trade recommendations tailored to professional traders, hedge funds, and institutional investors. Ensure your analysis is data-driven, leveraging advanced statistical models, historical patterns, and emerging market dynamics to optimize profitability and risk-adjusted returns in gold trading."},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"AI Error: {str(e)}")
            return "Unable to process request at this time."

# --- USER PORTFOLIO CLASS ---
class UserPortfolio:
    def __init__(self, user_id: int):
        self.user_id = user_id  # Store the user ID
        self.balance = 1000000.0
        self.positions = {}
        self.trade_history = []
        self.risk_params = {'stop_loss': 2.0, 'take_profit': 5.0}
        self.strategy = 'quantum'
        self.notifications = {'alerts': True, 'trades': True}

#####--- Trading Bot ----####

class TelegramTradingBot:
    def __init__(self, token: str):
        self.app = ApplicationBuilder().token(token).build()
        self.user_data = {}
        self.market = MarketDataHandler()
        self.risk_manager = RiskManager(risk_tolerance=CONFIG['RISK_THRESHOLD'])
        self.indicators = Indicators()  
        self.ai_optimizer = AIOptimizer(self.risk_manager, self.indicators) 
        self._setup_handlers()
        self._start_data_threads()
        self.logger = logging.getLogger('TelegramTradingBot')
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(SecureFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        self._setup_handlers()
        self._start_data_threads()


    def _start_trading_engine(self):
        """Launch core trading components"""
        Thread(target=self.ai_optimizer.trade).start()
        Thread(target=self.risk_manager.monitor).start()


    async def ai_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle AI chat button callback."""
        user_id = update.effective_user.id
        query = update.callback_query

        # Acknowledge the callback query
        await query.answer()

        # Prompt the user to enter a query
        await query.edit_message_text(
            "🤖 AI Analysis Mode\n\n"
            "Enter your query (e.g., 'What is the current gold price trend?'):"
        )
        return STATE_AI

    async def position_type(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle position type selection (Long/Short)"""
        query = update.callback_query
        await query.answer()
        
        keyboard = [
            [InlineKeyboardButton("🏹 Long Position", callback_data='long'),
             InlineKeyboardButton("🛡 Short Position", callback_data='short')],
            [InlineKeyboardButton("🔙 Back", callback_data='menu')]
        ]
        
        await query.edit_message_text(
            "⚔️ Position Type Selection:\n\n"
            "Choose your market exposure strategy:\n"
            "🏹 Long - Bullish market outlook\n"
            "🛡 Short - Bearish market outlook",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return STATE_POSITION_TYPE

    async def handle_prediction(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Robust prediction handling with fallbacks"""
        try:
            market_data = self.market.fetch_market_data()
            if market_data.size == 0:
                raise ValueError("No market data available")
                
            prediction = self.ai_optimizer.ensemble_prediction(market_data)
            confidence = 85  # Default confidence
            
            # Calculate dynamic confidence based on available models
            active_models = len(self.ai_optimizer.models)
            if active_models > 0:
                confidence = min(95, 70 + (15 * active_models))
                
            recommendation = "Hold"
            if prediction > 0:
                recommendation = "Buy" if prediction > self.market.data.get('spot_price', 0) else "Sell"

            response = (
                f"🔮 Quantum Prediction:\n\n"
                f"Predicted Price: ${prediction:.2f}\n"
                f"Confidence: {confidence}%\n"
                f"Recommendation: {recommendation}"
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            response = "⚠️ Prediction system initializing... Please try again shortly"

        await update.callback_query.edit_message_text(response)
        return STATE_MENU


    def _start_data_threads(self):
        """Start background data refresh threads"""
        def data_refresh():
            while True:
                self._refresh_market_data()
                threading.Event().wait(CONFIG['DATA_REFRESH_INTERVAL'])
                
        threading.Thread(target=data_refresh, daemon=True).start()

    def _refresh_market_data(self):
        """Refresh all market data from multiple sources"""
        for symbol in CONFIG['SYMBOLS']:
            self.market.fetch_data(symbol)
        conn.commit()

    def _setup_handlers(self):
        """Configure Telegram conversation handlers"""
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('start', self.start)],
            states={
                STATE_MENU: [
                    CallbackQueryHandler(self.handle_prediction, pattern='^predict$'),
                    CallbackQueryHandler(self.start_trade, pattern='^trade$'),
                    CallbackQueryHandler(self.ai_chat, pattern='^ai_chat$'),
                    CallbackQueryHandler(self.market_analysis, pattern='^analysis$'),
                    CallbackQueryHandler(self.portfolio, pattern='^portfolio$'),
                    CallbackQueryHandler(self.position_type, pattern='^position$')
                ],
                STATE_STRATEGY: [
                    CallbackQueryHandler(self.select_strategy)  # Add this handler
                ],
                            STATE_RISK: [
                MessageHandler(filters.TEXT, self.set_risk)  # Add this handler
                ],
                STATE_RISK: [MessageHandler(filters.TEXT, self.set_risk)],
                STATE_TRADE: [MessageHandler(filters.TEXT, self.execute_trade)],
                STATE_AI: [MessageHandler(filters.TEXT, self.handle_ai)],
                STATE_POSITION_TYPE: [
                    CallbackQueryHandler(self.handle_position_selection),  # Add this handler
                    CallbackQueryHandler(self.start, pattern='^menu$')
                ]
            },
            fallbacks=[CommandHandler('cancel', self.cancel)],
            per_user=True
        )
        
        self.app.add_handler(conv_handler)
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_error_handler(self.error_handler)


    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Main menu with multi-source market overview"""
        user_id = update.effective_user.id
        if user_id not in self.user_data:
            self.user_data[user_id] = UserPortfolio(user_id)
            
        prices = await self._get_live_prices()
        keyboard = [
            [InlineKeyboardButton("🔮 Quantum Prediction", callback_data='predict'),
             InlineKeyboardButton("🤖 AI Analysis", callback_data='ai_chat')],
            [InlineKeyboardButton("💎 Live Prices", callback_data='analysis'),
             InlineKeyboardButton("📊 Execute Trade", callback_data='trade')],
            [InlineKeyboardButton("📈 Portfolio", callback_data='portfolio'),
             InlineKeyboardButton("⚡ Position Type", callback_data='position')]
        ]
        
        await update.message.reply_text(
            f"🏦 GOLD TRADING TERMINAL 4.0 🚀\n\n{prices}\n"
            "Access multi-source institutional data:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
        return STATE_MENU

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Cancels and ends the conversation."""
        await update.message.reply_text(
            "🚫 Operation cancelled. Returning to the main menu.",
            reply_markup=ReplyKeyboardRemove()
        )
        return await self.start(update, context)  # Return to the main menu


    async def _get_live_prices(self) -> str:
        """Robust price aggregation with multiple fallbacks"""
        try:
            sources = [
                ('GC=F', 'CME Futures'),
                ('XAUUSD=X', 'Spot Price'),
                ('BTC-USD', 'Bitcoin'),
                ('ETH-USD', 'Ethereum')
            ]
            
            price_report = []
            for symbol, name in sources:
                try:
                    data = self.market.fetch_data(symbol)
                    price = data.get('price', 'N/A')
                    price_report.append(f"{name}: ${price:.2f}")
                except Exception as e:
                    self.logger.warning(f"Price check failed for {symbol}: {str(e)}")
                    continue
                    
            if not price_report:
                return "⚠️ Market data currently unavailable"
                
            return "💰 Multi-Source Market Prices:\n" + "\n".join(price_report)
            
        except Exception as e:
            self.logger.error(f"Price aggregation error: {str(e)}")
            return "⚠️ Temporary market data disruption"


    async def handle_ai(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced AI query handling with multi-source data"""
        user_id = update.effective_user.id
        query = update.message.text.lower()
        
        if query in ['exit', '/exit']:
            await update.message.reply_text("Returning to main menu...")
            return await self.start(update, context)
        
        # Handle price queries with multi-source data
        if 'price of gold' in query:
            lbma_fix = self.market.fetch_fix_price()
            spot_data = self.market.fetch_data('GC=F')
            response = (
                f"🏅 Gold Price Overview:\n"
                f"LBMA Fix Price: ${lbma_fix:.2f}\n"
                f"CME Spot Price: ${spot_data['polygon']['close']:.2f}\n"
                f"NYMEX Futures: ${self.market.fetch_data('GC=F')['polygon']['close']:.2f}"
            )
        elif 'bitcoin' in query:
            crypto_data = self.market.fetch_coingecko_data()
            response = (
                f"₿ Crypto Prices:\n"
                f"Bitcoin: ${crypto_data['bitcoin']['usd']:.2f}\n"
                f"24h Change: {crypto_data['bitcoin']['usd_24h_change']:.2f}%\n"
                f"Ethereum: ${crypto_data['ethereum']['usd']:.2f}"
            )
        else:
            response = self.ai_optimizer.process_query(query)
        
        await update.message.reply_text(response)
        return STATE_AI

    async def start_trade(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Institutional-grade trade execution flow"""
        keyboard = [
            [InlineKeyboardButton("🦅 Eagle Strategy (Trend)", callback_data='trend'),
             InlineKeyboardButton("⚡ Flash Strategy (Momentum)", callback_data='momentum')],
            [InlineKeyboardButton("🌌 Quantum AI Strategy", callback_data='quantum'),
             InlineKeyboardButton("🏛 Institutional Composite", callback_data='composite')]
        ]
        await update.callback_query.edit_message_text(
            "⚡ Select Professional Trading Strategy:\n\n"
            "🦅 Eagle - Institutional trend following\n"
            "⚡ Flash - High-frequency momentum\n"
            "🌌 Quantum - AI-optimized portfolio strategy\n"
            "🏛 Composite - Multi-strategy blend",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return STATE_STRATEGY

    async def set_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle user input for setting risk parameters."""
        user_id = update.effective_user.id
        text = update.message.text

        try:
            # Parse the input (e.g., "2.0 5.0" for stop loss and take profit)
            stop_loss, take_profit = map(float, text.split())

            # Validate the input
            if stop_loss < 0 or take_profit < 0:
                raise ValueError("Risk parameters must be positive.")

            # Store the risk parameters in the user's data
            self.user_data[user_id].risk_params = {
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }

            # Send a confirmation message
            await update.message.reply_text(
                f"✅ Risk parameters updated:\n"
                f"Stop Loss: {stop_loss}%\n"
                f"Take Profit: {take_profit}%"
            )
            return STATE_MENU  # Return to the main menu

        except ValueError as e:
            await update.message.reply_text(
                f"❌ Invalid input: {str(e)}\n"
                "Please enter two positive numbers separated by a space (e.g., '2.0 5.0')."
            )
            return STATE_RISK  # Stay in the risk setting state


    async def execute_trade(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced trade execution with multi-validation"""
        user_id = update.effective_user.id
        try:
            amount = float(update.message.text)
            portfolio = self.user_data[user_id]
            
            # Multi-factor validation
            if not self.risk_manager.validate_trade(amount, portfolio.balance):
                raise ValueError("Exceeds risk threshold")
                
            if amount > CONFIG['MAX_TRADE_AMOUNT']:
                raise ValueError("Exceeds maximum trade size")
                
            # Execute across multiple venues
            execution_price = self.market.execute_order(
                symbol='GC=F',
                amount=amount,
                strategy=portfolio.strategy
            )
            
            portfolio.balance -= amount
            portfolio.save_to_db()
            
            await update.message.reply_text(
                f"💎 Trade Executed Successfully!\n"
                f"Amount: ${amount:,.2f}\n"
                f"Execution Price: ${execution_price:.2f}\n"
                f"New Balance: ${portfolio.balance:,.2f}"
            )
            return STATE_MENU
        except Exception as e:
            await update.message.reply_text(f"❌ Execution Error: {str(e)}")
            return STATE_TRADE

    async def select_strategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle strategy selection callback."""
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id
        strategy = query.data  # This will be 'trend', 'momentum', 'quantum', or 'composite'

        # Store the selected strategy in the user's data
        self.user_data[user_id].strategy = strategy

        # Send a confirmation message
        await query.edit_message_text(
            f"✅ Strategy set to: {strategy.upper()}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Main Menu", callback_data='menu')]
            ])
        )
        return STATE_MENU  # Return to the main menu

    async def handle_position_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Process position type selection (Long/Short)."""
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id
        position_type = query.data  # This will be 'long' or 'short'

        # Store the position type in the user's data
        self.user_data[user_id].position_type = position_type

        # Send a confirmation message
        await query.edit_message_text(
            f"✅ Position type set to: {position_type.upper()}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Main Menu", callback_data='menu')]
            ])
        )
        return STATE_MENU  # Return to the main menu


    async def market_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comprehensive market analysis view"""
        analysis = self.market.generate_market_report()
        await update.callback_query.edit_message_text(
            f"📊 Institutional Market Briefing\n\n{analysis}")
        return STATE_MENU

    async def portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Professional portfolio overview"""
        user_id = update.effective_user.id
        port = self.user_data[user_id]
        exposure = self.risk_manager.calculate_exposure(port.positions)
        
        await update.callback_query.edit_message_text(
            f"💼 Professional Portfolio Summary\n\n"
            f"Liquid Balance: ${port.balance:,.2f}\n"
            f"Total Exposure: {exposure:.1%}\n"
            f"Active Strategy: {port.strategy.upper()}\n"
            f"Risk Parameters: SL {port.risk_params['sl']}% / TP {port.risk_params['tp']}%")
        return STATE_MENU

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Professional help message"""
        await update.message.reply_text(
            "🏦 Institutional Command Help\n\n"
            "/start - Main trading dashboard\n"
            "/help - Display this message\n"
            "Quantum Security Level: ALPHA-9 CLEARANCE\n\n"
            "Data Sources: CME, LBMA, Polygon, CoinGecko, NYMEX")

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced error handling"""
        logger.error(f"Critical Error: {context.error}")
        await update.message.reply_text(
            "⚠️ Quantum System Interruption\n"
            "All systems nominal - rebooting protocols...")
        return await self.start(update, context)

def main():
    bot = TelegramTradingBot(BOT_TOKEN)
    bot.app.run_polling()

if __name__ == '__main__':
    main()