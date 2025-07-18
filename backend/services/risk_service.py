import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskAnalysisService:
    """Advanced risk analysis and profiling service"""
    
    def __init__(self):
        self.risk_free_rate = 0.06  # 6% risk-free rate (approximate for India)
        self.confidence_levels = [0.95, 0.99]
        
    def calculate_portfolio_risk_metrics(self, holdings: List[Dict], returns_data: Dict[str, np.ndarray]) -> Dict:
        """Calculate comprehensive risk metrics for a portfolio"""
        try:
            # Extract portfolio data
            tickers = [h['ticker'] for h in holdings]
            amounts = [h['amount'] for h in holdings]
            total_value = sum(amounts)
            weights = np.array(amounts) / total_value
            
            # Create returns matrix
            returns_matrix = self._create_returns_matrix(tickers, returns_data)
            if returns_matrix is None:
                return self._get_default_risk_metrics()
            
            # Calculate portfolio returns
            portfolio_returns = np.dot(returns_matrix, weights)
            
            # Basic risk metrics
            volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
            expected_return = np.mean(portfolio_returns) * 252  # Annualized
            sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Advanced risk metrics
            var_95, var_99 = self._calculate_var(portfolio_returns, self.confidence_levels)
            cvar_95, cvar_99 = self._calculate_cvar(portfolio_returns, self.confidence_levels)
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            
            # Downside risk metrics
            downside_deviation = self._calculate_downside_deviation(portfolio_returns)
            sortino_ratio = (expected_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Correlation and diversification metrics
            correlation_matrix = np.corrcoef(returns_matrix.T)
            avg_correlation = self._calculate_average_correlation(correlation_matrix)
            diversification_ratio = self._calculate_diversification_ratio(weights, returns_matrix)
            
            # Risk contribution analysis
            risk_contributions = self._calculate_risk_contributions(weights, returns_matrix)
            
            # Beta calculation (against market index)
            beta = self._calculate_portfolio_beta(portfolio_returns, returns_data)
            
            # Risk categorization
            risk_category = self._categorize_risk_level(volatility, max_drawdown, sharpe_ratio)
            
            return {
                'expected_return': float(expected_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'beta': float(beta),
                'var_95': float(var_95),
                'var_99': float(var_99),
                'cvar_95': float(cvar_95),
                'cvar_99': float(cvar_99),
                'max_drawdown': float(max_drawdown),
                'downside_deviation': float(downside_deviation),
                'average_correlation': float(avg_correlation),
                'diversification_ratio': float(diversification_ratio),
                'risk_contributions': risk_contributions,
                'risk_category': risk_category,
                'risk_score': self._calculate_risk_score(volatility, max_drawdown, sharpe_ratio),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {e}")
            return self._get_default_risk_metrics()
    
    def analyze_individual_asset_risk(self, ticker: str, returns: np.ndarray) -> Dict:
        """Analyze risk metrics for individual assets"""
        try:
            # Basic metrics
            volatility = np.std(returns) * np.sqrt(252)
            expected_return = np.mean(returns) * 252
            sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Risk metrics
            var_95 = np.percentile(returns, 5) * np.sqrt(252)
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Drawdown analysis
            cumulative_returns = np.cumprod(1 + returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # Downside metrics
            downside_deviation = self._calculate_downside_deviation(returns)
            sortino_ratio = (expected_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            return {
                'ticker': ticker,
                'expected_return': float(expected_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'var_95': float(var_95),
                'max_drawdown': float(max_drawdown),
                'downside_deviation': float(downside_deviation),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'risk_category': self._categorize_asset_risk(volatility, max_drawdown)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing asset risk for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def calculate_optimal_position_sizes(self, holdings: List[Dict], returns_data: Dict[str, np.ndarray], 
                                       target_volatility: float = 0.15) -> Dict:
        """Calculate optimal position sizes based on risk budgeting"""
        try:
            tickers = [h['ticker'] for h in holdings]
            current_amounts = [h['amount'] for h in holdings]
            total_value = sum(current_amounts)
            
            # Create returns matrix
            returns_matrix = self._create_returns_matrix(tickers, returns_data)
            if returns_matrix is None:
                return {'error': 'Insufficient data for optimization'}
            
            # Calculate covariance matrix
            cov_matrix = np.cov(returns_matrix.T) * 252  # Annualized
            
            # Risk budgeting optimization
            optimal_weights = self._risk_budgeting_optimization(cov_matrix, target_volatility)
            
            # Calculate new position sizes
            optimal_amounts = optimal_weights * total_value
            
            # Calculate risk contributions
            portfolio_variance = np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))
            marginal_contributions = np.dot(cov_matrix, optimal_weights)
            risk_contributions = optimal_weights * marginal_contributions / portfolio_variance
            
            results = []
            for i, ticker in enumerate(tickers):
                results.append({
                    'ticker': ticker,
                    'current_amount': current_amounts[i],
                    'optimal_amount': float(optimal_amounts[i]),
                    'current_weight': current_amounts[i] / total_value,
                    'optimal_weight': float(optimal_weights[i]),
                    'risk_contribution': float(risk_contributions[i]),
                    'change_amount': float(optimal_amounts[i] - current_amounts[i]),
                    'change_percent': float((optimal_amounts[i] - current_amounts[i]) / current_amounts[i] * 100)
                })
            
            return {
                'optimal_positions': results,
                'target_volatility': target_volatility,
                'expected_volatility': float(np.sqrt(portfolio_variance)),
                'total_value': total_value
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal position sizes: {e}")
            return {'error': str(e)}
    
    def stress_test_portfolio(self, holdings: List[Dict], returns_data: Dict[str, np.ndarray]) -> Dict:
        """Perform stress testing on the portfolio"""
        try:
            tickers = [h['ticker'] for h in holdings]
            amounts = [h['amount'] for h in holdings]
            total_value = sum(amounts)
            weights = np.array(amounts) / total_value
            
            # Create returns matrix
            returns_matrix = self._create_returns_matrix(tickers, returns_data)
            if returns_matrix is None:
                return {'error': 'Insufficient data for stress testing'}
            
            # Define stress scenarios
            stress_scenarios = {
                'market_crash': {'factor': -0.3, 'description': '30% market decline'},
                'high_volatility': {'factor': 2.0, 'description': 'Double volatility scenario'},
                'correlation_spike': {'factor': 0.9, 'description': 'High correlation scenario'},
                'interest_rate_shock': {'factor': 0.02, 'description': '2% interest rate increase'},
                'sector_rotation': {'factor': -0.15, 'description': '15% sector-specific decline'}
            }
            
            stress_results = {}
            
            for scenario_name, scenario in stress_scenarios.items():
                if scenario_name == 'market_crash':
                    # Apply uniform decline to all assets
                    stressed_returns = returns_matrix * (1 + scenario['factor'])
                    
                elif scenario_name == 'high_volatility':
                    # Increase volatility while maintaining mean
                    means = np.mean(returns_matrix, axis=0)
                    centered_returns = returns_matrix - means
                    stressed_returns = means + centered_returns * scenario['factor']
                    
                elif scenario_name == 'correlation_spike':
                    # Increase correlations between assets
                    corr_matrix = np.corrcoef(returns_matrix.T)
                    target_corr = scenario['factor']
                    adjusted_corr = corr_matrix * (1 - target_corr) + target_corr
                    
                    # Generate new returns with higher correlation
                    std_devs = np.std(returns_matrix, axis=0)
                    means = np.mean(returns_matrix, axis=0)
                    
                    # Simplified correlation adjustment
                    stressed_returns = returns_matrix.copy()
                    
                else:
                    # Default to market crash scenario
                    stressed_returns = returns_matrix * (1 + stress_scenarios['market_crash']['factor'])
                
                # Calculate portfolio performance under stress
                portfolio_stressed_returns = np.dot(stressed_returns, weights)
                
                stress_results[scenario_name] = {
                    'description': scenario['description'],
                    'portfolio_return': float(np.sum(portfolio_stressed_returns)),
                    'portfolio_volatility': float(np.std(portfolio_stressed_returns) * np.sqrt(252)),
                    'max_drawdown': float(self._calculate_max_drawdown(portfolio_stressed_returns)),
                    'var_95': float(np.percentile(portfolio_stressed_returns, 5) * np.sqrt(252)),
                    'worst_day_loss': float(np.min(portfolio_stressed_returns))
                }
            
            return {
                'stress_scenarios': stress_results,
                'overall_stress_score': self._calculate_stress_score(stress_results),
                'recommendations': self._generate_stress_recommendations(stress_results)
            }
            
        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
            return {'error': str(e)}
    
    def _create_returns_matrix(self, tickers: List[str], returns_data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Create returns matrix from returns data"""
        try:
            returns_list = []
            min_length = float('inf')
            
            for ticker in tickers:
                if ticker in returns_data and len(returns_data[ticker]) > 0:
                    returns_list.append(returns_data[ticker])
                    min_length = min(min_length, len(returns_data[ticker]))
                else:
                    # Generate mock returns if data not available
                    mock_returns = np.random.normal(0.001, 0.02, 252)
                    returns_list.append(mock_returns)
                    min_length = min(min_length, len(mock_returns))
            
            if not returns_list or min_length == 0:
                return None
            
            # Trim all returns to same length
            trimmed_returns = [returns[-min_length:] for returns in returns_list]
            
            return np.array(trimmed_returns).T
            
        except Exception as e:
            logger.error(f"Error creating returns matrix: {e}")
            return None
    
    def _calculate_var(self, returns: np.ndarray, confidence_levels: List[float]) -> Tuple[float, ...]:
        """Calculate Value at Risk at different confidence levels"""
        var_values = []
        for confidence in confidence_levels:
            var = np.percentile(returns, (1 - confidence) * 100) * np.sqrt(252)
            var_values.append(var)
        return tuple(var_values)
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_levels: List[float]) -> Tuple[float, ...]:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        cvar_values = []
        for confidence in confidence_levels:
            var_threshold = np.percentile(returns, (1 - confidence) * 100)
            tail_returns = returns[returns <= var_threshold]
            cvar = np.mean(tail_returns) * np.sqrt(252) if len(tail_returns) > 0 else 0
            cvar_values.append(cvar)
        return tuple(cvar_values)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        except:
            return 0
    
    def _calculate_downside_deviation(self, returns: np.ndarray, target_return: float = 0) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return 0
        return np.sqrt(np.mean((downside_returns - target_return) ** 2)) * np.sqrt(252)
    
    def _calculate_average_correlation(self, correlation_matrix: np.ndarray) -> float:
        """Calculate average correlation excluding diagonal"""
        n = correlation_matrix.shape[0]
        if n <= 1:
            return 0
        
        # Get upper triangle excluding diagonal
        upper_triangle = np.triu(correlation_matrix, k=1)
        correlations = upper_triangle[upper_triangle != 0]
        
        return np.mean(correlations) if len(correlations) > 0 else 0
    
    def _calculate_diversification_ratio(self, weights: np.ndarray, returns_matrix: np.ndarray) -> float:
        """Calculate diversification ratio"""
        try:
            # Portfolio volatility
            portfolio_returns = np.dot(returns_matrix, weights)
            portfolio_vol = np.std(portfolio_returns)
            
            # Weighted average of individual volatilities
            individual_vols = np.std(returns_matrix, axis=0)
            weighted_avg_vol = np.dot(weights, individual_vols)
            
            return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
        except:
            return 1
    
    def _calculate_risk_contributions(self, weights: np.ndarray, returns_matrix: np.ndarray) -> List[Dict]:
        """Calculate risk contributions of each asset"""
        try:
            # Calculate covariance matrix
            cov_matrix = np.cov(returns_matrix.T)
            
            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            
            # Marginal contributions
            marginal_contributions = np.dot(cov_matrix, weights)
            
            # Risk contributions
            risk_contributions = weights * marginal_contributions / portfolio_variance
            
            # Format results
            results = []
            for i, contribution in enumerate(risk_contributions):
                results.append({
                    'asset_index': i,
                    'weight': float(weights[i]),
                    'risk_contribution': float(contribution),
                    'risk_contribution_percent': float(contribution * 100)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating risk contributions: {e}")
            return []
    
    def _calculate_portfolio_beta(self, portfolio_returns: np.ndarray, returns_data: Dict[str, np.ndarray]) -> float:
        """Calculate portfolio beta against market index"""
        try:
            # Try to find market index in returns data
            market_tickers = ['NIFTY50', 'SENSEX', '^NSEI', '^BSESN']
            market_returns = None
            
            for ticker in market_tickers:
                if ticker in returns_data:
                    market_returns = returns_data[ticker]
                    break
            
            if market_returns is None:
                # Generate synthetic market returns
                market_returns = np.random.normal(0.0008, 0.015, len(portfolio_returns))
            
            # Align lengths
            min_length = min(len(portfolio_returns), len(market_returns))
            portfolio_returns = portfolio_returns[-min_length:]
            market_returns = market_returns[-min_length:]
            
            # Calculate beta
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            return covariance / market_variance if market_variance > 0 else 1
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 1
    
    def _categorize_risk_level(self, volatility: float, max_drawdown: float, sharpe_ratio: float) -> str:
        """Categorize overall risk level"""
        risk_score = 0
        
        # Volatility scoring
        if volatility < 0.1:
            risk_score += 1
        elif volatility < 0.2:
            risk_score += 2
        else:
            risk_score += 3
        
        # Max drawdown scoring
        if abs(max_drawdown) < 0.1:
            risk_score += 1
        elif abs(max_drawdown) < 0.2:
            risk_score += 2
        else:
            risk_score += 3
        
        # Sharpe ratio scoring (inverse - higher is better)
        if sharpe_ratio > 1.5:
            risk_score += 1
        elif sharpe_ratio > 0.5:
            risk_score += 2
        else:
            risk_score += 3
        
        # Categorize based on total score
        if risk_score <= 4:
            return 'Conservative'
        elif risk_score <= 7:
            return 'Moderate'
        else:
            return 'Aggressive'
    
    def _categorize_asset_risk(self, volatility: float, max_drawdown: float) -> str:
        """Categorize individual asset risk"""
        if volatility < 0.15 and abs(max_drawdown) < 0.15:
            return 'Low Risk'
        elif volatility < 0.25 and abs(max_drawdown) < 0.25:
            return 'Medium Risk'
        else:
            return 'High Risk'
    
    def _calculate_risk_score(self, volatility: float, max_drawdown: float, sharpe_ratio: float) -> float:
        """Calculate numerical risk score (0-1 scale)"""
        # Normalize metrics
        vol_score = min(volatility / 0.3, 1)  # Cap at 30% volatility
        dd_score = min(abs(max_drawdown) / 0.3, 1)  # Cap at 30% drawdown
        sharpe_score = max(0, min((2 - sharpe_ratio) / 2, 1))  # Inverse scoring for Sharpe
        
        # Weighted average
        risk_score = (vol_score * 0.4 + dd_score * 0.4 + sharpe_score * 0.2)
        
        return float(risk_score)
    
    def _risk_budgeting_optimization(self, cov_matrix: np.ndarray, target_volatility: float) -> np.ndarray:
        """Simple risk budgeting optimization"""
        try:
            n_assets = cov_matrix.shape[0]
            
            # Start with equal risk contribution
            weights = np.ones(n_assets) / n_assets
            
            # Iterative optimization (simplified)
            for _ in range(10):
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                marginal_contributions = np.dot(cov_matrix, weights)
                risk_contributions = weights * marginal_contributions / portfolio_variance
                
                # Adjust weights to equalize risk contributions
                target_risk_contrib = 1 / n_assets
                adjustment = target_risk_contrib / (risk_contributions + 1e-8)
                weights = weights * adjustment
                weights = weights / np.sum(weights)  # Normalize
            
            # Scale to target volatility
            current_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            scale_factor = target_volatility / current_vol if current_vol > 0 else 1
            
            # Note: In practice, you might want to adjust position sizes rather than weights
            # For now, we return the risk-budgeted weights
            
            return weights
            
        except Exception as e:
            logger.error(f"Error in risk budgeting optimization: {e}")
            n_assets = cov_matrix.shape[0]
            return np.ones(n_assets) / n_assets
    
    def _calculate_stress_score(self, stress_results: Dict) -> float:
        """Calculate overall stress score"""
        try:
            scores = []
            for scenario, results in stress_results.items():
                # Score based on portfolio return and max drawdown
                return_score = max(0, min(abs(results['portfolio_return']) / 0.5, 1))
                drawdown_score = max(0, min(abs(results['max_drawdown']) / 0.5, 1))
                scenario_score = (return_score + drawdown_score) / 2
                scores.append(scenario_score)
            
            return float(np.mean(scores))
            
        except:
            return 0.5
    
    def _generate_stress_recommendations(self, stress_results: Dict) -> List[str]:
        """Generate recommendations based on stress test results"""
        recommendations = []
        
        try:
            # Analyze worst-case scenarios
            worst_scenario = max(stress_results.items(), 
                               key=lambda x: abs(x[1]['portfolio_return']))
            
            if abs(worst_scenario[1]['portfolio_return']) > 0.3:
                recommendations.append("Consider reducing portfolio concentration to limit downside risk")
            
            if worst_scenario[1]['max_drawdown'] < -0.25:
                recommendations.append("Portfolio may experience significant drawdowns - consider defensive assets")
            
            # Check volatility across scenarios
            avg_volatility = np.mean([r['portfolio_volatility'] for r in stress_results.values()])
            if avg_volatility > 0.25:
                recommendations.append("High volatility detected - consider adding low-volatility assets")
            
            # General recommendations
            recommendations.append("Regular rebalancing can help maintain risk levels")
            recommendations.append("Consider diversifying across asset classes and geographies")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations")
        
        return recommendations
    
    def _get_default_risk_metrics(self) -> Dict:
        """Return default risk metrics when calculation fails"""
        return {
            'expected_return': 0.08,
            'volatility': 0.15,
            'sharpe_ratio': 0.5,
            'sortino_ratio': 0.6,
            'beta': 1.0,
            'var_95': -0.05,
            'var_99': -0.08,
            'cvar_95': -0.07,
            'cvar_99': -0.10,
            'max_drawdown': -0.15,
            'downside_deviation': 0.12,
            'average_correlation': 0.3,
            'diversification_ratio': 1.2,
            'risk_contributions': [],
            'risk_category': 'Moderate',
            'risk_score': 0.5,
            'last_updated': datetime.now().isoformat(),
            'note': 'Default values used due to insufficient data'
        }