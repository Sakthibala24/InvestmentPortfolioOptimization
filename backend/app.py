from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
import json
from services.csv_data_service import CSVDataService

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///portfolio.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
CORS(app)
db = SQLAlchemy(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize CSV data service
csv_data_service = CSVDataService()

# Database Models
class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    holdings = db.relationship('Holding', backref='portfolio', lazy=True, cascade='all, delete-orphan')

class Holding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolio.id'), nullable=False)
    ticker = db.Column(db.String(20), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    asset_class = db.Column(db.String(50), nullable=False)
    current_price = db.Column(db.Float)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class OptimizationResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolio.id'), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    strategy = db.Column(db.String(50), nullable=False)
    risk_level = db.Column(db.Float, nullable=False)
    results = db.Column(db.Text, nullable=False)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Portfolio Optimization Engine
class PortfolioOptimizer:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': None,  # Would implement XGBoost if available
            'LSTM': None,     # Would implement LSTM if TensorFlow available
            'Transformer': None  # Would implement Transformer if available
        }
        self.scaler = StandardScaler()
    
    def get_market_data(self, tickers, period='1y'):
        """Fetch market data for given tickers"""
        try:
            # Use CSV data service instead of yfinance
            for ticker in tickers:
                # Handle Indian stock tickers
                if not ticker.endswith('.NS') and ticker not in ['CASH', 'GOLDETF']:
                    ticker_symbol = f"{ticker}.NS"
                else:
                    ticker_symbol = ticker
                
                stock_data = csv_data_service.get_stock_data(ticker_symbol, period)
                if stock_data and stock_data.get('historical_data'):
                    prices = [float(d['close']) for d in stock_data['historical_data']]
                    data[ticker] = np.array(prices)
                else:
                    data[ticker] = self._generate_mock_data()
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {ticker: self._generate_mock_data() for ticker in tickers}
    
    def _generate_mock_data(self, length=252):
        """Generate mock price data for testing"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, length)
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        return np.array(prices[1:])
    
    def calculate_returns(self, price_data):
        """Calculate daily returns from price data"""
        returns = {}
        for ticker, prices in price_data.items():
            if len(prices) > 1:
                daily_returns = np.diff(prices) / prices[:-1]
                returns[ticker] = daily_returns
            else:
                returns[ticker] = np.array([0.001])  # Default small positive return
        return returns
    
    def modern_portfolio_theory(self, returns_data, risk_level):
        """Implement Modern Portfolio Theory optimization"""
        tickers = list(returns_data.keys())
        returns_matrix = np.array([returns_data[ticker] for ticker in tickers]).T
        
        # Calculate expected returns and covariance matrix
        expected_returns = np.mean(returns_matrix, axis=0)
        cov_matrix = np.cov(returns_matrix.T)
        
        # Simple optimization based on risk level
        n_assets = len(tickers)
        if risk_level < 0.3:  # Conservative
            weights = np.ones(n_assets) / n_assets  # Equal weights
            weights = weights * 0.8  # Reduce exposure
        elif risk_level < 0.7:  # Moderate
            # Weight by inverse volatility
            volatilities = np.sqrt(np.diag(cov_matrix))
            weights = (1 / volatilities) / np.sum(1 / volatilities)
        else:  # Aggressive
            # Weight by expected returns
            weights = expected_returns / np.sum(expected_returns)
            weights = np.maximum(weights, 0)  # No short selling
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return dict(zip(tickers, weights))
    
    def optimize_portfolio(self, holdings, model_type, strategy, risk_level):
        """Main optimization function"""
        try:
            tickers = [h['ticker'] for h in holdings]
            amounts = [h['amount'] for h in holdings]
            
            # Get market data
            price_data = self.get_market_data(tickers)
            returns_data = self.calculate_returns(price_data)
            
            # Apply optimization strategy
            if strategy == 'MPT':
                weights = self.modern_portfolio_theory(returns_data, risk_level)
            elif strategy == 'Black-Litterman':
                weights = self._black_litterman_optimization(returns_data, risk_level)
            elif strategy == 'Risk Parity':
                weights = self._risk_parity_optimization(returns_data)
            else:  # Hybrid
                weights = self._hybrid_optimization(returns_data, risk_level)
            
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
                    'change_percent': ((optimized_amount - holding['amount']) / holding['amount']) * 100
                })
            
            return {
                'success': True,
                'optimized_holdings': optimized_holdings,
                'total_value': total_value,
                'expected_return': self._calculate_expected_return(weights, returns_data),
                'risk_metrics': self._calculate_risk_metrics(weights, returns_data)
            }
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _black_litterman_optimization(self, returns_data, risk_level):
        """Simplified Black-Litterman implementation"""
        tickers = list(returns_data.keys())
        n_assets = len(tickers)
        
        # Market cap weights (simplified)
        market_weights = np.ones(n_assets) / n_assets
        
        # Adjust based on risk level
        if risk_level > 0.7:
            # Increase concentration for aggressive investors
            market_weights = market_weights ** 0.5
            market_weights = market_weights / np.sum(market_weights)
        
        return dict(zip(tickers, market_weights))
    
    def _risk_parity_optimization(self, returns_data):
        """Risk parity optimization"""
        tickers = list(returns_data.keys())
        returns_matrix = np.array([returns_data[ticker] for ticker in tickers]).T
        
        # Calculate volatilities
        volatilities = np.std(returns_matrix, axis=0)
        
        # Inverse volatility weights
        weights = (1 / volatilities) / np.sum(1 / volatilities)
        
        return dict(zip(tickers, weights))
    
    def _hybrid_optimization(self, returns_data, risk_level):
        """Hybrid optimization combining multiple strategies"""
        mpt_weights = self.modern_portfolio_theory(returns_data, risk_level)
        rp_weights = self._risk_parity_optimization(returns_data)
        
        # Blend weights based on risk level
        alpha = risk_level  # Higher risk level favors MPT
        
        tickers = list(returns_data.keys())
        hybrid_weights = {}
        
        for ticker in tickers:
            hybrid_weights[ticker] = (alpha * mpt_weights[ticker] + 
                                    (1 - alpha) * rp_weights[ticker])
        
        return hybrid_weights
    
    def _calculate_expected_return(self, weights, returns_data):
        """Calculate portfolio expected return"""
        expected_returns = {}
        for ticker, returns in returns_data.items():
            expected_returns[ticker] = np.mean(returns) * 252  # Annualized
        
        portfolio_return = sum(weights[ticker] * expected_returns[ticker] 
                             for ticker in weights.keys())
        return portfolio_return
    
    def _calculate_risk_metrics(self, weights, returns_data):
        """Calculate portfolio risk metrics"""
        tickers = list(weights.keys())
        returns_matrix = np.array([returns_data[ticker] for ticker in tickers]).T
        
        # Portfolio returns
        weights_array = np.array([weights[ticker] for ticker in tickers])
        portfolio_returns = np.dot(returns_matrix, weights_array)
        
        # Risk metrics
        volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        
        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': self._calculate_expected_return(weights, returns_data) / volatility if volatility > 0 else 0
        }
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

# Initialize optimizer
optimizer = PortfolioOptimizer()

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

@app.route('/api/portfolio', methods=['POST'])
def create_portfolio():
    """Create a new portfolio"""
    try:
        data = request.get_json()
        
        portfolio = Portfolio(
            user_id=data.get('user_id', 'default'),
            name=data.get('name', 'My Portfolio')
        )
        
        db.session.add(portfolio)
        db.session.flush()  # Get the ID
        
        # Add holdings
        for holding_data in data.get('holdings', []):
            holding = Holding(
                portfolio_id=portfolio.id,
                ticker=holding_data['ticker'],
                amount=holding_data['amount'],
                asset_class=holding_data['asset_class']
            )
            db.session.add(holding)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'portfolio_id': portfolio.id,
            'message': 'Portfolio created successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating portfolio: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/portfolio/<int:portfolio_id>', methods=['GET'])
def get_portfolio(portfolio_id):
    """Get portfolio details"""
    try:
        portfolio = Portfolio.query.get_or_404(portfolio_id)
        
        holdings_data = []
        for holding in portfolio.holdings:
            holdings_data.append({
                'id': holding.id,
                'ticker': holding.ticker,
                'amount': holding.amount,
                'asset_class': holding.asset_class,
                'current_price': holding.current_price,
                'last_updated': holding.last_updated.isoformat() if holding.last_updated else None
            })
        
        return jsonify({
            'success': True,
            'portfolio': {
                'id': portfolio.id,
                'name': portfolio.name,
                'created_at': portfolio.created_at.isoformat(),
                'updated_at': portfolio.updated_at.isoformat(),
                'holdings': holdings_data
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize_portfolio():
    """Optimize portfolio using AI models"""
    try:
        data = request.get_json()
        
        holdings = data.get('holdings', [])
        model_type = data.get('model', 'Random Forest')
        strategy = data.get('strategy', 'MPT')
        risk_level = data.get('risk_level', 0.5)
        
        # Run optimization
        result = optimizer.optimize_portfolio(holdings, model_type, strategy, risk_level)
        
        if result['success']:
            # Save optimization result if portfolio_id provided
            portfolio_id = data.get('portfolio_id')
            if portfolio_id:
                opt_result = OptimizationResult(
                    portfolio_id=portfolio_id,
                    model_type=model_type,
                    strategy=strategy,
                    risk_level=risk_level,
                    results=json.dumps(result)  # Convert to JSON string
                )
                db.session.add(opt_result)
                db.session.commit()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/market-data/<ticker>', methods=['GET'])
def get_market_data(ticker):
    """Get market data for a specific ticker"""
    try:
        period = request.args.get('period', '1y')
        
        # Use CSV data service
        stock_data = csv_data_service.get_stock_data(ticker, period)
        
        if not stock_data:
            return jsonify({'success': False, 'error': 'No data found for ticker'})
            
        return jsonify({
            'success': True,
            'ticker': ticker,
            'current_price': stock_data['current_price'],
            'volatility': stock_data['volatility'],
            'annual_return': stock_data['annual_return'],
            'sharpe_ratio': stock_data['sharpe_ratio'],
            'price_history': {d['date']: d['close'] for d in stock_data['historical_data']}
        })
        
    except Exception as e:
        logger.error(f"Error fetching market data for {ticker}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/risk-profile', methods=['POST'])
def calculate_risk_profile():
    """Calculate risk profile for given portfolio"""
    try:
        data = request.get_json()
        holdings = data.get('holdings', [])
        
        if not holdings:
            return jsonify({'success': False, 'error': 'No holdings provided'})
        
        tickers = [h['ticker'] for h in holdings]
        amounts = [h['amount'] for h in holdings]
        weights = np.array(amounts) / sum(amounts)
        
        # Get market data and calculate metrics
        price_data = csv_data_service.get_returns_data(tickers)
        returns_data = optimizer.calculate_returns(price_data)
        
        # Calculate portfolio metrics
        portfolio_metrics = optimizer._calculate_risk_metrics(
            dict(zip(tickers, weights)), returns_data
        )
        
        # Risk categorization
        volatility = portfolio_metrics['volatility']
        if volatility < 0.1:
            risk_category = 'Conservative'
        elif volatility < 0.2:
            risk_category = 'Moderate'
        else:
            risk_category = 'Aggressive'
        
        return jsonify({
            'success': True,
            'risk_profile': {
                'category': risk_category,
                'volatility': float(portfolio_metrics['volatility']),
                'max_drawdown': float(portfolio_metrics['max_drawdown']),
                'sharpe_ratio': float(portfolio_metrics['sharpe_ratio']),
                'expected_return': float(optimizer._calculate_expected_return(
                    dict(zip(tickers, weights)), returns_data
                ))
            }
        })
        
    except Exception as e:
        logger.error(f"Error calculating risk profile: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/benchmark', methods=['GET'])
def get_benchmark_data():
    """Get benchmark comparison data"""
    try:
        # Use sector performance from CSV data
        sector_data = csv_data_service.get_sector_performance()
        
        benchmark_data = []
        for name, data in sector_data.items():
            if data:
                benchmark_data.append({
                    'name': name,
                    'annual_return': round(data['annual_return'], 2),
                    'volatility': round(data['volatility'], 2),
                    'sharpe_ratio': round(data['sharpe_ratio'], 2),
                    'max_drawdown': round(data['max_drawdown'], 2)
                })
        
        return jsonify({
            'success': True,
            'benchmarks': benchmark_data
        })
        
    except Exception as e:
        logger.error(f"Error fetching benchmark data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/portfolio/<int:portfolio_id>/update', methods=['PUT'])
def update_portfolio(portfolio_id):
    """Update portfolio holdings"""
    try:
        data = request.get_json()
        portfolio = Portfolio.query.get_or_404(portfolio_id)
        
        # Update portfolio name if provided
        if 'name' in data:
            portfolio.name = data['name']
        
        # Update holdings
        if 'holdings' in data:
            # Remove existing holdings
            Holding.query.filter_by(portfolio_id=portfolio_id).delete()
            
            # Add new holdings
            for holding_data in data['holdings']:
                holding = Holding(
                    portfolio_id=portfolio_id,
                    ticker=holding_data['ticker'],
                    amount=holding_data['amount'],
                    asset_class=holding_data['asset_class']
                )
                db.session.add(holding)
        
        portfolio.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Portfolio updated successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating portfolio: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# Initialize database
@app.before_first_request
def create_tables():
    db.create_all()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    app.run(debug=True, host='0.0.0.0', port=5000)