import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AdvancedPortfolioOptimizer:
    """Advanced portfolio optimization service with multiple AI models"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': None,  # Placeholder for XGBoost
            'LSTM': None,     # Placeholder for LSTM
            'Transformer': None  # Placeholder for Transformer
        }
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def prepare_features(self, price_data, returns_data):
        """Prepare features for ML models"""
        features = {}
        
        for ticker in price_data.keys():
            prices = price_data[ticker]
            returns = returns_data[ticker]
            
            # Technical indicators
            sma_5 = pd.Series(prices).rolling(5).mean().fillna(method='bfill')
            sma_20 = pd.Series(prices).rolling(20).mean().fillna(method='bfill')
            rsi = self._calculate_rsi(prices)
            volatility = pd.Series(returns).rolling(20).std().fillna(method='bfill')
            
            # Price momentum
            momentum_5 = pd.Series(prices).pct_change(5).fillna(0)
            momentum_20 = pd.Series(prices).pct_change(20).fillna(0)
            
            features[ticker] = {
                'sma_5': sma_5.iloc[-1] if len(sma_5) > 0 else prices[-1],
                'sma_20': sma_20.iloc[-1] if len(sma_20) > 0 else prices[-1],
                'rsi': rsi,
                'volatility': volatility.iloc[-1] if len(volatility) > 0 else np.std(returns),
                'momentum_5': momentum_5.iloc[-1] if len(momentum_5) > 0 else 0,
                'momentum_20': momentum_20.iloc[-1] if len(momentum_20) > 0 else 0,
                'current_price': prices[-1],
                'avg_return': np.mean(returns),
                'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            }
        
        return features
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        prices = pd.Series(prices)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 else 50
    
    def train_ml_model(self, historical_data, model_type='Random Forest'):
        """Train machine learning model for return prediction"""
        try:
            # Prepare training data
            X, y = self._prepare_training_data(historical_data)
            
            if len(X) < 10:  # Not enough data for training
                logger.warning("Insufficient data for ML training, using fallback method")
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            model = self.models.get(model_type)
            if model is None:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Model {model_type} - MSE: {mse:.4f}, R2: {r2:.4f}")
            
            return {
                'model': model,
                'scaler': self.scaler,
                'mse': mse,
                'r2': r2,
                'feature_columns': self.feature_columns
            }
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            return None
    
    def _prepare_training_data(self, historical_data):
        """Prepare training data from historical price data"""
        X, y = [], []
        
        for ticker, data in historical_data.items():
            prices = data['prices']
            returns = data['returns']
            
            # Create features for each time window
            for i in range(30, len(prices) - 5):  # Need enough history and future data
                # Features (technical indicators at time i)
                window_prices = prices[i-30:i]
                window_returns = returns[i-30:i]
                
                features = [
                    np.mean(window_prices[-5:]) / np.mean(window_prices[-20:]),  # SMA ratio
                    np.std(window_returns),  # Volatility
                    np.mean(window_returns),  # Average return
                    (prices[i] - prices[i-5]) / prices[i-5],  # 5-day momentum
                    (prices[i] - prices[i-20]) / prices[i-20],  # 20-day momentum
                ]
                
                # Target (future 5-day return)
                future_return = (prices[i+5] - prices[i]) / prices[i]
                
                X.append(features)
                y.append(future_return)
        
        self.feature_columns = [
            'sma_ratio', 'volatility', 'avg_return', 
            'momentum_5d', 'momentum_20d'
        ]
        
        return np.array(X), np.array(y)
    
    def predict_returns(self, trained_model, current_features):
        """Predict future returns using trained model"""
        try:
            if trained_model is None:
                return self._fallback_return_prediction(current_features)
            
            model = trained_model['model']
            scaler = trained_model['scaler']
            
            # Prepare feature matrix
            feature_matrix = []
            for ticker, features in current_features.items():
                feature_row = [
                    features['sma_5'] / features['sma_20'] if features['sma_20'] > 0 else 1,
                    features['volatility'],
                    features['avg_return'],
                    features['momentum_5'],
                    features['momentum_20']
                ]
                feature_matrix.append(feature_row)
            
            # Scale and predict
            X_scaled = scaler.transform(feature_matrix)
            predictions = model.predict(X_scaled)
            
            # Return predictions as dictionary
            tickers = list(current_features.keys())
            return dict(zip(tickers, predictions))
            
        except Exception as e:
            logger.error(f"Error predicting returns: {e}")
            return self._fallback_return_prediction(current_features)
    
    def _fallback_return_prediction(self, current_features):
        """Fallback return prediction based on simple heuristics"""
        predictions = {}
        for ticker, features in current_features.items():
            # Simple prediction based on momentum and volatility
            momentum_score = (features['momentum_5'] + features['momentum_20']) / 2
            volatility_penalty = features['volatility'] * 0.5
            predicted_return = momentum_score - volatility_penalty
            predictions[ticker] = predicted_return
        
        return predictions
    
    def optimize_with_ml(self, holdings, model_type, strategy, risk_level):
        """Optimize portfolio using machine learning predictions"""
        try:
            tickers = [h['ticker'] for h in holdings]
            amounts = [h['amount'] for h in holdings]
            
            # Get historical data for training
            historical_data = self._get_historical_data(tickers, period='2y')
            
            # Train ML model
            trained_model = self.train_ml_model(historical_data, model_type)
            
            # Get current market data
            current_price_data = self._get_current_market_data(tickers)
            current_returns_data = self._calculate_returns(current_price_data)
            
            # Prepare features
            current_features = self.prepare_features(current_price_data, current_returns_data)
            
            # Predict future returns
            predicted_returns = self.predict_returns(trained_model, current_features)
            
            # Optimize based on predictions and strategy
            if strategy == 'MPT':
                weights = self._ml_enhanced_mpt(predicted_returns, current_returns_data, risk_level)
            elif strategy == 'Black-Litterman':
                weights = self._ml_enhanced_black_litterman(predicted_returns, current_returns_data, risk_level)
            elif strategy == 'Risk Parity':
                weights = self._ml_enhanced_risk_parity(predicted_returns, current_returns_data)
            else:  # Hybrid
                weights = self._ml_enhanced_hybrid(predicted_returns, current_returns_data, risk_level)
            
            # Calculate optimized amounts
            total_value = sum(amounts)
            optimized_holdings = []
            
            for i, holding in enumerate(holdings):
                ticker = holding['ticker']
                weight = weights.get(ticker, 1/len(holdings))
                optimized_amount = total_value * weight
                
                optimized_holdings.append({
                    'ticker': ticker,
                    'current_amount': holding['amount'],
                    'optimized_amount': optimized_amount,
                    'weight': weight,
                    'predicted_return': predicted_returns.get(ticker, 0),
                    'change_percent': ((optimized_amount - holding['amount']) / holding['amount']) * 100
                })
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(weights, current_returns_data, predicted_returns)
            
            return {
                'success': True,
                'optimized_holdings': optimized_holdings,
                'total_value': total_value,
                'model_performance': {
                    'mse': trained_model['mse'] if trained_model else None,
                    'r2': trained_model['r2'] if trained_model else None
                },
                'portfolio_metrics': portfolio_metrics
            }
            
        except Exception as e:
            logger.error(f"Error in ML optimization: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_historical_data(self, tickers, period='2y'):
        """Get historical data for model training"""
        historical_data = {}
        
        for ticker in tickers:
            try:
                # Handle Indian stock tickers
                if not ticker.endswith('.NS') and ticker not in ['CASH', 'GOLDETF']:
                    ticker_symbol = f"{ticker}.NS"
                else:
                    ticker_symbol = ticker
                
                stock = yf.Ticker(ticker_symbol)
                hist = stock.history(period=period)
                
                if not hist.empty:
                    prices = hist['Close'].values
                    returns = np.diff(prices) / prices[:-1]
                    
                    historical_data[ticker] = {
                        'prices': prices,
                        'returns': returns,
                        'volume': hist['Volume'].values,
                        'dates': hist.index
                    }
                else:
                    # Generate mock data
                    historical_data[ticker] = self._generate_mock_historical_data()
                    
            except Exception as e:
                logger.warning(f"Error fetching historical data for {ticker}: {e}")
                historical_data[ticker] = self._generate_mock_historical_data()
        
        return historical_data
    
    def _get_current_market_data(self, tickers):
        """Get current market data"""
        current_data = {}
        
        for ticker in tickers:
            try:
                if not ticker.endswith('.NS') and ticker not in ['CASH', 'GOLDETF']:
                    ticker_symbol = f"{ticker}.NS"
                else:
                    ticker_symbol = ticker
                
                stock = yf.Ticker(ticker_symbol)
                hist = stock.history(period='3mo')  # Last 3 months for current analysis
                
                if not hist.empty:
                    current_data[ticker] = hist['Close'].values
                else:
                    current_data[ticker] = self._generate_mock_data(length=60)
                    
            except Exception as e:
                logger.warning(f"Error fetching current data for {ticker}: {e}")
                current_data[ticker] = self._generate_mock_data(length=60)
        
        return current_data
    
    def _calculate_returns(self, price_data):
        """Calculate returns from price data"""
        returns_data = {}
        for ticker, prices in price_data.items():
            if len(prices) > 1:
                returns_data[ticker] = np.diff(prices) / prices[:-1]
            else:
                returns_data[ticker] = np.array([0.001])
        return returns_data
    
    def _generate_mock_historical_data(self, length=500):
        """Generate mock historical data for testing"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, length)
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices[1:])
        returns = np.diff(prices) / prices[:-1]
        
        return {
            'prices': prices,
            'returns': returns,
            'volume': np.random.randint(1000, 10000, length),
            'dates': pd.date_range(end=datetime.now(), periods=length, freq='D')
        }
    
    def _generate_mock_data(self, length=60):
        """Generate mock price data"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, length)
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        return np.array(prices[1:])
    
    def _ml_enhanced_mpt(self, predicted_returns, returns_data, risk_level):
        """ML-enhanced Modern Portfolio Theory"""
        tickers = list(predicted_returns.keys())
        
        # Combine historical and predicted returns
        combined_returns = {}
        for ticker in tickers:
            historical_return = np.mean(returns_data[ticker]) * 252  # Annualized
            predicted_return = predicted_returns[ticker] * 252  # Annualized
            
            # Weight based on confidence (simplified)
            alpha = 0.7  # Weight for predictions
            combined_returns[ticker] = alpha * predicted_return + (1 - alpha) * historical_return
        
        # Calculate weights based on expected returns and risk level
        returns_array = np.array(list(combined_returns.values()))
        
        if risk_level < 0.3:  # Conservative
            weights = np.ones(len(tickers)) / len(tickers)  # Equal weights
        elif risk_level < 0.7:  # Moderate
            # Weight by predicted returns, but cap maximum allocation
            weights = np.maximum(returns_array, 0)
            weights = weights / np.sum(weights)
            weights = np.minimum(weights, 0.4)  # Max 40% in any asset
            weights = weights / np.sum(weights)  # Renormalize
        else:  # Aggressive
            # Weight heavily by predicted returns
            weights = np.maximum(returns_array, 0) ** 2
            weights = weights / np.sum(weights)
        
        return dict(zip(tickers, weights))
    
    def _ml_enhanced_black_litterman(self, predicted_returns, returns_data, risk_level):
        """ML-enhanced Black-Litterman optimization"""
        tickers = list(predicted_returns.keys())
        n_assets = len(tickers)
        
        # Start with market cap weights (simplified as equal weights)
        market_weights = np.ones(n_assets) / n_assets
        
        # Adjust based on ML predictions
        prediction_adjustment = np.array(list(predicted_returns.values()))
        prediction_adjustment = prediction_adjustment / np.sum(np.abs(prediction_adjustment))
        
        # Blend market weights with predictions
        alpha = risk_level  # Higher risk level gives more weight to predictions
        adjusted_weights = (1 - alpha) * market_weights + alpha * np.abs(prediction_adjustment)
        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
        
        return dict(zip(tickers, adjusted_weights))
    
    def _ml_enhanced_risk_parity(self, predicted_returns, returns_data):
        """ML-enhanced Risk Parity optimization"""
        tickers = list(predicted_returns.keys())
        
        # Calculate volatilities
        volatilities = []
        for ticker in tickers:
            vol = np.std(returns_data[ticker]) * np.sqrt(252)  # Annualized
            volatilities.append(vol)
        
        volatilities = np.array(volatilities)
        
        # Inverse volatility weights
        inv_vol_weights = (1 / volatilities) / np.sum(1 / volatilities)
        
        # Adjust based on predicted returns
        return_adjustment = np.array(list(predicted_returns.values()))
        return_adjustment = (return_adjustment - np.min(return_adjustment)) / (np.max(return_adjustment) - np.min(return_adjustment) + 1e-8)
        
        # Combine risk parity with return predictions
        combined_weights = inv_vol_weights * (1 + return_adjustment)
        combined_weights = combined_weights / np.sum(combined_weights)
        
        return dict(zip(tickers, combined_weights))
    
    def _ml_enhanced_hybrid(self, predicted_returns, returns_data, risk_level):
        """ML-enhanced Hybrid optimization"""
        # Get weights from different strategies
        mpt_weights = self._ml_enhanced_mpt(predicted_returns, returns_data, risk_level)
        bl_weights = self._ml_enhanced_black_litterman(predicted_returns, returns_data, risk_level)
        rp_weights = self._ml_enhanced_risk_parity(predicted_returns, returns_data)
        
        tickers = list(predicted_returns.keys())
        
        # Blend strategies based on risk level
        if risk_level < 0.3:  # Conservative - favor risk parity
            weights = {ticker: 0.6 * rp_weights[ticker] + 0.3 * bl_weights[ticker] + 0.1 * mpt_weights[ticker] 
                      for ticker in tickers}
        elif risk_level < 0.7:  # Moderate - balanced approach
            weights = {ticker: 0.4 * rp_weights[ticker] + 0.3 * bl_weights[ticker] + 0.3 * mpt_weights[ticker] 
                      for ticker in tickers}
        else:  # Aggressive - favor MPT
            weights = {ticker: 0.1 * rp_weights[ticker] + 0.3 * bl_weights[ticker] + 0.6 * mpt_weights[ticker] 
                      for ticker in tickers}
        
        return weights
    
    def _calculate_portfolio_metrics(self, weights, returns_data, predicted_returns):
        """Calculate comprehensive portfolio metrics"""
        tickers = list(weights.keys())
        weights_array = np.array([weights[ticker] for ticker in tickers])
        
        # Historical metrics
        returns_matrix = np.array([returns_data[ticker] for ticker in tickers]).T
        portfolio_returns = np.dot(returns_matrix, weights_array)
        
        historical_return = np.mean(portfolio_returns) * 252  # Annualized
        volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = historical_return / volatility if volatility > 0 else 0
        
        # Predicted metrics
        predicted_portfolio_return = sum(weights[ticker] * predicted_returns[ticker] * 252 
                                       for ticker in tickers)
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)  # 95% VaR
        
        return {
            'historical_return': float(historical_return),
            'predicted_return': float(predicted_portfolio_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'var_95': float(var_95),
            'diversification_ratio': self._calculate_diversification_ratio(weights_array, returns_matrix)
        }
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_diversification_ratio(self, weights, returns_matrix):
        """Calculate diversification ratio"""
        try:
            # Portfolio volatility
            portfolio_returns = np.dot(returns_matrix, weights)
            portfolio_vol = np.std(portfolio_returns)
            
            # Weighted average of individual volatilities
            individual_vols = np.std(returns_matrix, axis=0)
            weighted_avg_vol = np.dot(weights, individual_vols)
            
            # Diversification ratio
            div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
            return float(div_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {e}")
            return 1.0