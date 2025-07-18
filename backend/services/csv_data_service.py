import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class CSVDataService:
    """Service for loading and processing data from stock_data_long.csv"""
    
    def __init__(self):
        self.data = None
        self.cache = {}
        self.csv_path = Path(__file__).parent.parent / 'stock_data_long.csv'
        self.load_data()
        
    def load_data(self):
        """Load data from CSV file"""
        try:
            if not self.csv_path.exists():
                logger.error(f"CSV file not found at {self.csv_path}")
                return
                
            logger.info(f"Loading data from {self.csv_path}")
            self.data = pd.read_csv(self.csv_path)
            
            # Convert Date column to datetime
            if 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'])
            
            # Get unique tickers
            if 'Ticker' in self.data.columns:
                self.available_tickers = self.data['Ticker'].unique().tolist()
                logger.info(f"Loaded data for {len(self.available_tickers)} tickers")
            else:
                logger.error("No 'Ticker' column found in CSV")
                
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            self.data = None
    
    def get_stock_data(self, ticker: str, period: str = '1y') -> Optional[Dict]:
        """Get comprehensive stock data for a ticker from CSV"""
        try:
            if self.data is None:
                logger.error("No data loaded")
                return None
                
            # Filter data for the specific ticker
            ticker_data = self.data[self.data['Ticker'] == ticker].copy()
            
            if ticker_data.empty:
                logger.warning(f"No data found for ticker {ticker}")
                return None
            
            # Sort by date
            ticker_data = ticker_data.sort_values('Date')
            
            # Filter by period
            end_date = ticker_data['Date'].max()
            if period == '1y':
                start_date = end_date - timedelta(days=365)
            elif period == '2y':
                start_date = end_date - timedelta(days=730)
            elif period == '6mo':
                start_date = end_date - timedelta(days=180)
            elif period == '3mo':
                start_date = end_date - timedelta(days=90)
            else:
                start_date = ticker_data['Date'].min()
            
            filtered_data = ticker_data[ticker_data['Date'] >= start_date]
            
            if filtered_data.empty:
                logger.warning(f"No data found for ticker {ticker} in period {period}")
                return None
            
            # Calculate basic metrics
            current_price = float(filtered_data['Close'].iloc[-1])
            previous_close = float(filtered_data['Close'].iloc[-2]) if len(filtered_data) > 1 else current_price
            
            # Calculate returns
            returns = filtered_data['Close'].pct_change().dropna()
            
            # Calculate technical indicators
            technical_data = self._calculate_technical_indicators(filtered_data)
            
            stock_data = {
                'ticker': ticker,
                'current_price': current_price,
                'previous_close': previous_close,
                'change': float(current_price - previous_close),
                'change_percent': float((current_price - previous_close) / previous_close * 100) if previous_close > 0 else 0,
                'volume': int(filtered_data['Volume'].iloc[-1]) if 'Volume' in filtered_data.columns else 0,
                'annual_return': float((current_price / filtered_data['Close'].iloc[0] - 1) * 100) if len(filtered_data) > 0 else 0,
                'volatility': float(returns.std() * np.sqrt(252) * 100) if len(returns) > 0 else 0,
                'max_drawdown': float(self._calculate_max_drawdown(filtered_data['Close'].values) * 100),
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                'historical_data': self._format_historical_data(filtered_data),
                'technical_indicators': technical_data,
                'last_updated': datetime.now().isoformat()
            }
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error getting stock data for {ticker}: {e}")
            return None
    
    def get_multiple_stocks_data(self, tickers: List[str], period: str = '1y') -> Dict[str, Dict]:
        """Get data for multiple stocks"""
        results = {}
        
        for ticker in tickers:
            data = self.get_stock_data(ticker, period)
            if data:
                results[ticker] = data
            else:
                # Generate fallback data
                results[ticker] = self._generate_fallback_data(ticker)
        
        return results
    
    def get_returns_data(self, tickers: List[str], period: str = '1y') -> Dict[str, np.ndarray]:
        """Get returns data for portfolio optimization"""
        returns_data = {}
        
        for ticker in tickers:
            try:
                if self.data is None:
                    returns_data[ticker] = self._generate_mock_returns()
                    continue
                    
                ticker_data = self.data[self.data['Ticker'] == ticker].copy()
                
                if ticker_data.empty:
                    returns_data[ticker] = self._generate_mock_returns()
                    continue
                
                # Sort by date and filter by period
                ticker_data = ticker_data.sort_values('Date')
                end_date = ticker_data['Date'].max()
                
                if period == '1y':
                    start_date = end_date - timedelta(days=365)
                elif period == '2y':
                    start_date = end_date - timedelta(days=730)
                else:
                    start_date = end_date - timedelta(days=365)
                
                filtered_data = ticker_data[ticker_data['Date'] >= start_date]
                
                if len(filtered_data) < 2:
                    returns_data[ticker] = self._generate_mock_returns()
                    continue
                
                # Calculate returns
                prices = filtered_data['Close'].values
                returns = np.diff(prices) / prices[:-1]
                returns_data[ticker] = returns
                
            except Exception as e:
                logger.error(f"Error getting returns for {ticker}: {e}")
                returns_data[ticker] = self._generate_mock_returns()
        
        return returns_data
    
    def get_available_tickers(self) -> List[str]:
        """Get list of available tickers in the CSV"""
        if self.data is not None and hasattr(self, 'available_tickers'):
            return self.available_tickers
        return []
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators from price data"""
        try:
            close_prices = data['Close']
            
            # Simple Moving Averages
            sma_5 = close_prices.rolling(window=5).mean()
            sma_20 = close_prices.rolling(window=20).mean()
            sma_50 = close_prices.rolling(window=50).mean()
            
            # RSI
            rsi = self._calculate_rsi(close_prices)
            
            # Volume indicators
            volume_sma = data['Volume'].rolling(window=20).mean() if 'Volume' in data.columns else None
            volume_ratio = data['Volume'].iloc[-1] / volume_sma.iloc[-1] if volume_sma is not None and volume_sma.iloc[-1] > 0 else 1
            
            return {
                'sma_5': float(sma_5.iloc[-1]) if not pd.isna(sma_5.iloc[-1]) else None,
                'sma_20': float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else None,
                'sma_50': float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None,
                'rsi': float(rsi) if not pd.isna(rsi) else None,
                'volume_ratio': float(volume_ratio) if volume_ratio else 1
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
            returns = np.diff(prices) / prices[:-1]
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        except:
            return 0
    
    def _format_historical_data(self, data: pd.DataFrame) -> List[Dict]:
        """Format historical data for API response"""
        try:
            formatted_data = []
            for _, row in data.tail(30).iterrows():  # Last 30 days
                formatted_data.append({
                    'date': row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else '',
                    'open': float(row['Open']) if 'Open' in row and pd.notna(row['Open']) else float(row['Close']),
                    'high': float(row['High']) if 'High' in row and pd.notna(row['High']) else float(row['Close']),
                    'low': float(row['Low']) if 'Low' in row and pd.notna(row['Low']) else float(row['Close']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']) if 'Volume' in row and pd.notna(row['Volume']) else 0
                })
            return formatted_data
        except Exception as e:
            logger.error(f"Error formatting historical data: {e}")
            return []
    
    def _generate_fallback_data(self, ticker: str) -> Dict:
        """Generate fallback data when ticker not found"""
        base_price = 100 + hash(ticker) % 1000
        return {
            'ticker': ticker,
            'current_price': float(base_price),
            'previous_close': float(base_price * 0.99),
            'change': float(base_price * 0.01),
            'change_percent': 1.0,
            'volume': 10000,
            'annual_return': 8.0,
            'volatility': 15.0,
            'max_drawdown': -12.0,
            'sharpe_ratio': 0.8,
            'historical_data': [],
            'technical_indicators': {},
            'last_updated': datetime.now().isoformat(),
            'note': 'Fallback data - ticker not found in CSV'
        }
    
    def _generate_mock_returns(self, length: int = 252) -> np.ndarray:
        """Generate mock returns for missing data"""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, length)
    
    def get_sector_performance(self) -> Dict[str, Dict]:
        """Get sector performance from available data"""
        if self.data is None:
            return {}
        
        # Group by sector if available, otherwise use sample tickers
        sector_data = {}
        sample_tickers = self.get_available_tickers()[:8]  # Take first 8 tickers as sectors
        
        for i, ticker in enumerate(sample_tickers):
            sector_name = f"Sector_{i+1}"
            stock_data = self.get_stock_data(ticker, '1y')
            
            if stock_data:
                sector_data[sector_name] = {
                    'current_value': stock_data['current_price'],
                    'annual_return': stock_data['annual_return'],
                    'volatility': stock_data['volatility'],
                    'sharpe_ratio': stock_data['sharpe_ratio'],
                    'max_drawdown': stock_data['max_drawdown']
                }
        
        return sector_data
    
    def get_market_status(self) -> Dict:
        """Get market status (simplified)"""
        now = datetime.now()
        market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_weekday = now.weekday() < 5
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
            'current_time': now.isoformat()
        }