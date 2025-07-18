from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Portfolio(db.Model):
    __tablename__ = 'portfolios'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    holdings = db.relationship('Holding', backref='portfolio', lazy=True, cascade='all, delete-orphan')
    optimizations = db.relationship('OptimizationResult', backref='portfolio', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'holdings': [holding.to_dict() for holding in self.holdings]
        }

class Holding(db.Model):
    __tablename__ = 'holdings'
    
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolios.id'), nullable=False)
    ticker = db.Column(db.String(20), nullable=False, index=True)
    amount = db.Column(db.Float, nullable=False)
    asset_class = db.Column(db.String(50), nullable=False)
    current_price = db.Column(db.Float)
    target_allocation = db.Column(db.Float)  # Target percentage allocation
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'ticker': self.ticker,
            'amount': self.amount,
            'asset_class': self.asset_class,
            'current_price': self.current_price,
            'target_allocation': self.target_allocation,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }

class OptimizationResult(db.Model):
    __tablename__ = 'optimization_results'
    
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolios.id'), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    strategy = db.Column(db.String(50), nullable=False)
    risk_level = db.Column(db.Float, nullable=False)
    results = db.Column(db.Text, nullable=False)  # JSON string
    performance_metrics = db.Column(db.Text)  # JSON string for metrics
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def get_results(self):
        return json.loads(self.results) if self.results else {}
    
    def set_results(self, results_dict):
        self.results = json.dumps(results_dict)
    
    def get_performance_metrics(self):
        return json.loads(self.performance_metrics) if self.performance_metrics else {}
    
    def set_performance_metrics(self, metrics_dict):
        self.performance_metrics = json.dumps(metrics_dict)
    
    def to_dict(self):
        return {
            'id': self.id,
            'model_type': self.model_type,
            'strategy': self.strategy,
            'risk_level': self.risk_level,
            'results': self.get_results(),
            'performance_metrics': self.get_performance_metrics(),
            'created_at': self.created_at.isoformat()
        }

class MarketData(db.Model):
    __tablename__ = 'market_data'
    
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), nullable=False, index=True)
    date = db.Column(db.Date, nullable=False, index=True)
    open_price = db.Column(db.Float)
    high_price = db.Column(db.Float)
    low_price = db.Column(db.Float)
    close_price = db.Column(db.Float, nullable=False)
    volume = db.Column(db.BigInteger)
    adjusted_close = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('ticker', 'date', name='unique_ticker_date'),)
    
    def to_dict(self):
        return {
            'ticker': self.ticker,
            'date': self.date.isoformat(),
            'open': self.open_price,
            'high': self.high_price,
            'low': self.low_price,
            'close': self.close_price,
            'volume': self.volume,
            'adjusted_close': self.adjusted_close
        }

class RiskProfile(db.Model):
    __tablename__ = 'risk_profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolios.id'), nullable=False)
    risk_score = db.Column(db.Float, nullable=False)  # 0-1 scale
    risk_category = db.Column(db.String(20), nullable=False)  # Conservative, Moderate, Aggressive
    volatility = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    beta = db.Column(db.Float)
    var_95 = db.Column(db.Float)  # Value at Risk 95%
    expected_return = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'risk_score': self.risk_score,
            'risk_category': self.risk_category,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'beta': self.beta,
            'var_95': self.var_95,
            'expected_return': self.expected_return,
            'created_at': self.created_at.isoformat()
        }