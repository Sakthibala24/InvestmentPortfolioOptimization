import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
from typing import Dict, List, Optional, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class MarketDataService:
    """Service for fetching and processing market data"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        self.indian_exchanges = {
            'NSE': '.NS',
            'BSE': '.BO'
        }
        
    def get_stock_data(self, ticker: str, period: str = '1y', exchange: str = 'NSE') -> Optional[Dict]:
        """Get comprehensive stock data for a ticker"""
        try:
            # Check cache first
            cache_key = f"{ticker}_{period}_{exchange}"
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            # Format ticker for Indian exchanges
            formatted_ticker = self._format_ticker(ticker, exchange)
            
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(formatted_ticker)
            
            # Get historical data
            hist = stock.history(period=period)
            if hist.empty:
                logger.warning(f"No historical data found for {ticker}")
                return None
            
            # Get stock info
            try:
                info = stock.info
            except:
                info = {}
            
            # Calculate technical indicators
            technical_data = self._calculate_technical_indicators(hist)
            
            # Prepare comprehensive data
            stock_data = {
                'ticker': ticker,
                'formatted_ticker': formatted_ticker,
                'current_price': float(hist['Close'].iloc[-1]),
                'previous_close': float(hist['Close'].iloc[-2]) if len(hist) > 1 else float(hist['Close'].iloc[-1]),
                'change': float(hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) if len(hist) > 1 else 0,
                'change_percent': float((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0,
                'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'historical_data': self._format_historical_data(hist),
                'technical_indicators': technical_data,
                'last_updated': datetime.now().isoformat()
            }
            
            # Cache the data
            self._cache_data(cache_key, stock_data)
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {e}")
            return None
    
    def get_multiple_stocks_data(self, tickers: List[str], period: str = '1y', exchange: str = 'NSE') -> Dict[str, Dict]:
        """Get data for multiple stocks concurrently"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.get_stock_data, ticker, period, exchange): ticker 
                for ticker in tickers
            }
            
            # Collect results
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data:
                        results[ticker] = data
                except Exception as e:
                    logger.error(f"Error fetching data for {ticker}: {e}")
                    results[ticker] = None
        
        return results
    
    def get_index_data(self, index_name: str, period: str = '1y') -> Optional[Dict]:
        """Get data for market indices"""
        index_symbols = {
            'NIFTY50': '^NSEI',
            'SENSEX': '^BSESN',
            'BANKNIFTY': '^NSEBANK',
            'NIFTYIT': '^CNXIT',
            'NIFTYFMCG': '^CNXFMCG',
            'NIFTYPHARMA': '^CNXPHARMA',
            'NIFTYAUTO': '^CNXAUTO',
            'NIFTYMETAL': '^CNXMETAL',
            'NIFTYREALTY': '^CNXREALTY',
            'NIFTYENERGY': '^CNXENERGY'
        }
        
        symbol = index_symbols.get(index_name.upper())
        if not symbol:
            logger.error(f"Unknown index: {index_name}")
            return None
        
        try:
            # Check cache
            cache_key = f"index_{index_name}_{period}"
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            # Fetch data
            index = yf.Ticker(symbol)
            hist = index.history(period=period)
            
            if hist.empty:
                logger.warning(f"No data found for index {index_name}")
                return None
            
            # Calculate metrics
            returns = hist['Close'].pct_change().dropna()
            
            index_data = {
                'name': index_name,
                'symbol': symbol,
                'current_value': float(hist['Close'].iloc[-1]),
                'previous_close': float(hist['Close'].iloc[-2]) if len(hist) > 1 else float(hist['Close'].iloc[-1]),
                'change': float(hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) if len(hist) > 1 else 0,
                'change_percent': float((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0,
                'annual_return': float((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100),
                'volatility': float(returns.std() * np.sqrt(252) * 100),
                'max_drawdown': float(self._calculate_max_drawdown(hist['Close'].values) * 100),
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                'historical_data': self._format_historical_data(hist),
                'last_updated': datetime.now().isoformat()
            }
            
            # Cache the data
            self._cache_data(cache_key, index_data)
            
            return index_data
            
        except Exception as e:
            logger.error(f"Error fetching index data for {index_name}: {e}")
            return None
    
    def get_sector_performance(self) -> Dict[str, Dict]:
        """Get performance data for different sectors"""
        sector_indices = {
            'Banking': '^NSEBANK',
            'IT': '^CNXIT',
            'FMCG': '^CNXFMCG',
            'Pharma': '^CNXPHARMA',
            'Auto': '^CNXAUTO',
            'Metal': '^CNXMETAL',
            'Realty': '^CNXREALTY',
            'Energy': '^CNXENERGY'
        }
        
        sector_data = {}
        
        for sector, symbol in sector_indices.items():
            try:
                index = yf.Ticker(symbol)
                hist = index.history(period='1y')
                
                if not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    
                    sector_data[sector] = {
                        'current_value': float(hist['Close'].iloc[-1]),
                        'annual_return': float((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100),
                        'volatility': float(returns.std() * np.sqrt(252) * 100),
                        'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                        'max_drawdown': float(self._calculate_max_drawdown(hist['Close'].values) * 100)
                    }
            except Exception as e:
                logger.error(f"Error fetching sector data for {sector}: {e}")
        
        return sector_data
    
    def get_commodity_data(self, commodity: str = 'GOLD', period: str = '1y') -> Optional[Dict]:
        """Get commodity data"""
        commodity_symbols = {
            'GOLD': 'GC=F',
            'SILVER': 'SI=F',
            'CRUDE': 'CL=F',
            'COPPER': 'HG=F',
            'NATURAL_GAS': 'NG=F'
        }
        
        symbol = commodity_symbols.get(commodity.upper())
        if not symbol:
            # Try Indian commodity ETFs
            if commodity.upper() == 'GOLD':
                symbol = 'GOLDBEES.NS'
            else:
                logger.error(f"Unknown commodity: {commodity}")
                return None
        
        try:
            # Check cache
            cache_key = f"commodity_{commodity}_{period}"
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            # Fetch data
            commodity_ticker = yf.Ticker(symbol)
            hist = commodity_ticker.history(period=period)
            
            if hist.empty:
                logger.warning(f"No data found for commodity {commodity}")
                return None
            
            returns = hist['Close'].pct_change().dropna()
            
            commodity_data = {
                'name': commodity,
                'symbol': symbol,
                'current_price': float(hist['Close'].iloc[-1]),
                'annual_return': float((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100),
                'volatility': float(returns.std() * np.sqrt(252) * 100),
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                'max_drawdown': float(self._calculate_max_drawdown(hist['Close'].values) * 100),
                'historical_data': self._format_historical_data(hist),
                'last_updated': datetime.now().isoformat()
            }
            
            # Cache the data
            self._cache_data(cache_key, commodity_data)
            
            return commodity_data
            
        except Exception as e:
            logger.error(f"Error fetching commodity data for {commodity}: {e}")
            return None
    
    def _format_ticker(self, ticker: str, exchange: str = 'NSE') -> str:
        """Format ticker for Indian exchanges"""
        if ticker in ['CASH', 'GOLDETF', 'LIQUIDFUND']:
            return ticker
        
        if not ticker.endswith(('.NS', '.BO')):
            suffix = self.indian_exchanges.get(exchange, '.NS')
            return f"{ticker}{suffix}"
        
        return ticker
    
    def _calculate_technical_indicators(self, hist: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        try:
            close_prices = hist['Close']
            
            # Simple Moving Averages
            sma_5 = close_prices.rolling(window=5).mean()
            sma_20 = close_prices.rolling(window=20).mean()
            sma_50 = close_prices.rolling(window=50).mean()
            
            # Exponential Moving Averages
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            
            # MACD
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            macd_histogram = macd - macd_signal
            
            # RSI
            rsi = self._calculate_rsi(close_prices)
            
            # Bollinger Bands
            bb_middle = close_prices.rolling(window=20).mean()
            bb_std = close_prices.rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            # Volume indicators
            if 'Volume' in hist.columns:
                volume_sma = hist['Volume'].rolling(window=20).mean()
                volume_ratio = hist['Volume'].iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1
            else:
                volume_ratio = 1
            
            return {
                'sma_5': float(sma_5.iloc[-1]) if not pd.isna(sma_5.iloc[-1]) else None,
                'sma_20': float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else None,
                'sma_50': float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None,
                'ema_12': float(ema_12.iloc[-1]) if not pd.isna(ema_12.iloc[-1]) else None,
                'ema_26': float(ema_26.iloc[-1]) if not pd.isna(ema_26.iloc[-1]) else None,
                'macd': float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None,
                'macd_signal': float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else None,
                'macd_histogram': float(macd_histogram.iloc[-1]) if not pd.isna(macd_histogram.iloc[-1]) else None,
                'rsi': float(rsi) if not pd.isna(rsi) else None,
                'bb_upper': float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else None,
                'bb_middle': float(bb_middle.iloc[-1]) if not pd.isna(bb_middle.iloc[-1]) else None,
                'bb_lower': float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else None,
                'volume_ratio': float(volume_ratio)
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        except:
            return 50
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = np.cumprod(1 + np.diff(prices) / prices[:-1])
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        except:
            return 0
    
    def _format_historical_data(self, hist: pd.DataFrame) -> List[Dict]:
        """Format historical data for API response"""
        try:
            formatted_data = []
            for date, row in hist.iterrows():
                formatted_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else 0
                })
            return formatted_data[-30:]  # Return last 30 days
        except Exception as e:
            logger.error(f"Error formatting historical data: {e}")
            return []
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return (datetime.now() - cache_time).seconds < self.cache_duration
    
    def _cache_data(self, cache_key: str, data: Dict):
        """Cache data with timestamp"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
    
    def get_market_status(self) -> Dict:
        """Get current market status"""
        try:
            # Get NIFTY data to determine market status
            nifty = yf.Ticker('^NSEI')
            hist = nifty.history(period='2d')
            
            if len(hist) < 2:
                return {'status': 'unknown', 'message': 'Unable to determine market status'}
            
            # Check if market is open (simplified logic)
            now = datetime.now()
            market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            is_weekday = now.weekday() < 5  # Monday = 0, Sunday = 6
            is_market_hours = market_open_time <= now <= market_close_time
            
            if is_weekday and is_market_hours:
                status = 'open'
                message = 'Market is currently open'
            elif is_weekday and now < market_open_time:
                status = 'pre_market'
                message = 'Pre-market session'
            elif is_weekday and now > market_close_time:
                status = 'after_market'
                message = 'After-market session'
            else:
                status = 'closed'
                message = 'Market is closed'
            
            return {
                'status': status,
                'message': message,
                'next_open': market_open_time.isoformat() if status == 'closed' else None,
                'next_close': market_close_time.isoformat() if status == 'open' else None,
                'current_time': now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {'status': 'unknown', 'message': 'Unable to determine market status'}