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
from threading import Thread
import ib_insync  # Interactive Brokers API integration
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Load environment variables
load_dotenv()

class Indicators:
    @staticmethod
    def moving_average(data, window=50):
        if 'close' in data.columns:
            return data['close'].rolling(window=window).mean()
        else:
            raise ValueError("DataFrame does not contain 'close' column")

    @staticmethod
    def rsi(data, window=14):
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(data, fast_period=12, slow_period=26, signal_period=9):
        fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return macd_line, signal_line

    @staticmethod
    def bollinger_bands(data, window=20, num_std_dev=2):
        middle_band = data['close'].rolling(window=window).mean()
        rolling_std = data['close'].rolling(window=window).std()
        upper_band = middle_band + (rolling_std * num_std_dev)
        lower_band = middle_band - (rolling_std * num_std_dev)
        return upper_band, middle_band, lower_band

    @staticmethod
    def generate_trade_signal(data):
        sma_50 = Indicators.moving_average(data, window=50)
        sma_200 = Indicators.moving_average(data, window=200)
        macd_line, signal_line = Indicators.macd(data)

        if sma_50.iloc[-1] > sma_200.iloc[-1] and macd_line.iloc[-1] > signal_line.iloc[-1]:
            return "BUY"
        elif sma_50.iloc[-1] < sma_200.iloc[-1] and macd_line.iloc[-1] < signal_line.iloc[-1]:
            return "SELL"
        else:
            return "HOLD"

    @staticmethod
    def apply_all_indicators(data):
        sma_50 = Indicators.moving_average(data, window=50)
        sma_200 = Indicators.moving_average(data, window=200)
        rsi_14 = Indicators.rsi(data, window=14)
        macd_line, signal_line = Indicators.macd(data)
        upper_band, middle_band, lower_band = Indicators.bollinger_bands(data)

        data['SMA_50'] = sma_50
        data['SMA_200'] = sma_200
        data['RSI_14'] = rsi_14
        data['MACD'] = macd_line
        data['Signal_Line'] = signal_line
        data['Upper_Band'] = upper_band
        data['Middle_Band'] = middle_band
        data['Lower_Band'] = lower_band

        return data

class MarketDataHandler:
    def __init__(self, symbols: List[str], start_date: str, risk_tolerance: float = 0.5):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.data: Optional[Dict[str, pd.DataFrame]] = {}
        self.risk_tolerance = risk_tolerance
        self.POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
        self.IBKR_TWS_HOST = os.getenv('IBKR_TWS_HOST')
        self.IBKR_TWS_PORT = os.getenv('IBKR_TWS_PORT')

    def fetch_data(self, source: str = "yfinance") -> None:
        try:
            def fetch_from_source(symbol, source):
                if source == "yfinance":
                    return self.fetch_from_yfinance(symbol)
                elif source == "coingecko":
                    return self.fetch_from_coingecko(symbol)
                elif source == "polygon":
                    return self.fetch_from_polygon(symbol)
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
            if not self.POLYGON_API_KEY:
                raise ValueError("Polygon API key is missing.")
            client = RESTClient(self.POLYGON_API_KEY)

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

    def fetch_from_ibkr(self, symbol: str) -> None:
        try:
            logger.info(f"Fetching data from IBKR for {symbol}...")
            ibkr_client = ib_insync.IB()
            ibkr_client.connect(self.IBKR_TWS_HOST, self.IBKR_TWS_PORT, clientId=1)

            contract = ib_insync.Stock(symbol, 'SMART', 'USD')
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
                    df_with_indicators = Indicators.apply_all_indicators(df)
                    analysis_results[symbol] = {"close": df['close'].iloc[-1], "indicators": df_with_indicators.iloc[-1]}  # Add indicators here
                    logger.info(f"📈 Market Analysis for {symbol}: {analysis_results[symbol]}")
                except Exception as e:
                    logger.error(f"Error analyzing data for {symbol}: {e}")
            return analysis_results
        else:
            logger.warning(f"No market data available.")
            return {}

    def trade_decision(self) -> None:
        analysis_results = self.analyze_market_data()
        for symbol, data in analysis_results.items():
            signal = Indicators.generate_trade_signal(data['indicators'])
            logger.info(f"Trade decision for {symbol}: {signal}")

    def run(self):
        while True:
            self.fetch_data(source="yfinance")  # Automatically fetch data every 60 seconds
            self.trade_decision()
            time.sleep(60)

            # Fetch IBKR data every 20 seconds
            self.fetch_data(source="ibkr")
            self.trade_decision()
            time.sleep(20)


if __name__ == "__main__":
    symbols = ['GOLD', 'BTC', 'XAU-USD']  # You can add more symbols if needed
    start_date = '2000-01-01'
    market_data_handler = MarketDataHandler(symbols, start_date)

    # Run the bot for gold trading
    market_data_handler.run()


