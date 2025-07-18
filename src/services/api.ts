// API service for backend communication
const API_BASE_URL = 'http://localhost:5000/api';

export interface PortfolioItem {
  ticker: string;
  amount: number;
  assetClass: string;
  optimizedAmount?: number;
  weight?: number;
  predicted_return?: number;
  change_percent?: number;
}

export interface OptimizationRequest {
  holdings: PortfolioItem[];
  model: string;
  strategy: string;
  risk_level: number;
  portfolio_id?: number;
}

export interface OptimizationResult {
  success: boolean;
  optimized_holdings?: PortfolioItem[];
  total_value?: number;
  model_performance?: {
    mse?: number;
    r2?: number;
  };
  portfolio_metrics?: {
    expected_return: number;
    volatility: number;
    sharpe_ratio: number;
    max_drawdown: number;
  };
  error?: string;
}

export interface RiskProfile {
  success: boolean;
  risk_profile?: {
    category: string;
    volatility: number;
    max_drawdown: number;
    sharpe_ratio: number;
    expected_return: number;
  };
  error?: string;
}

export interface BenchmarkData {
  success: boolean;
  benchmarks?: Array<{
    name: string;
    annual_return: number;
    volatility: number;
    sharpe_ratio: number;
    max_drawdown: number;
  }>;
  error?: string;
}

export interface HealthCheck {
  status: string;
  timestamp: string;
}

class ApiService {
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  async healthCheck(): Promise<HealthCheck> {
    return this.request<HealthCheck>('/health');
  }

  async optimizePortfolio(data: OptimizationRequest): Promise<OptimizationResult> {
    return this.request<OptimizationResult>('/optimize', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async calculateRiskProfile(holdings: PortfolioItem[]): Promise<RiskProfile> {
    return this.request<RiskProfile>('/risk-profile', {
      method: 'POST',
      body: JSON.stringify({ holdings }),
    });
  }

  async getBenchmarkData(): Promise<BenchmarkData> {
    return this.request<BenchmarkData>('/benchmark');
  }

  async getMarketData(ticker: string, period: string = '1y') {
    return this.request(`/market-data/${ticker}?period=${period}`);
  }

  async createPortfolio(portfolioData: {
    user_id?: string;
    name: string;
    holdings: PortfolioItem[];
  }) {
    return this.request('/portfolio', {
      method: 'POST',
      body: JSON.stringify(portfolioData),
    });
  }

  async getPortfolio(portfolioId: number) {
    return this.request(`/portfolio/${portfolioId}`);
  }
}

export const apiService = new ApiService();