import React, { useState } from "react";
import { Bar, Pie } from "react-chartjs-2";
import "chart.js/auto";
import { TrendingUp, PieChart, BarChart3, Target, Wrench as Benchmark, Plus, Settings, Brain } from "lucide-react";

interface PortfolioItem {
  ticker: string;
  amount: number;
  assetClass: string;
  optimizedAmount?: number;
}

const App = () => {
  const [page, setPage] = useState("input");
  const [portfolio, setPortfolio] = useState<PortfolioItem[]>([
    { ticker: "RELIANCE", amount: 10000, assetClass: "Stock" },
    { ticker: "TCS", amount: 15000, assetClass: "Stock" },
    { ticker: "GOLDETF", amount: 5000, assetClass: "Gold" },
    { ticker: "CASH", amount: 2000, assetClass: "Cash" }
  ]);
  const [optimized, setOptimized] = useState<PortfolioItem[] | null>(null);
  const [model, setModel] = useState("Random Forest");
  const [strategy, setStrategy] = useState("MPT");
  const [risk, setRisk] = useState(0.5);

  const runOptimization = () => {
    const updated = portfolio.map((item) => ({
      ...item,
      optimizedAmount: item.amount * (1 + (0.15 - risk * 0.2))
    }));
    setOptimized(updated);
  };

  const tabs = [
    { value: "input", label: "Portfolio Input", icon: <TrendingUp className="w-4 h-4" /> },
    { value: "opt", label: "Optimization", icon: <Settings className="w-4 h-4" /> },
    { value: "viz", label: "Visualizations", icon: <PieChart className="w-4 h-4" /> },
    { value: "risk", label: "Risk Profiling", icon: <Brain className="w-4 h-4" /> },
    { value: "bench", label: "Benchmark", icon: <Benchmark className="w-4 h-4" /> }
  ];

  const renderTabs = () => (
    <div className="flex flex-wrap gap-2 mb-8 bg-white dark:bg-gray-800 p-2 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
      {tabs.map(({ value, label, icon }) => (
        <button
          key={value}
          className={`flex items-center gap-2 px-4 py-2 rounded-md font-medium transition-all duration-200 ${
            page === value
              ? "bg-blue-600 text-white shadow-md"
              : "text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-900 dark:hover:text-gray-200"
          }`}
          onClick={() => setPage(value)}
        >
          {icon}
          <span className="hidden sm:inline">{label}</span>
        </button>
      ))}
    </div>
  );

  const totalValue = portfolio.reduce((sum, item) => sum + item.amount, 0);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2 flex items-center justify-center gap-3">
            <TrendingUp className="w-10 h-10 text-blue-600" />
            Portfolio Optimizer Dashboard
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Advanced portfolio optimization with AI-powered insights
          </p>
        </div>

        {renderTabs()}

        {page === "input" && (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-6">
              <TrendingUp className="w-6 h-6 text-blue-600" />
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Manual Portfolio Entry</h2>
            </div>
            
            <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-blue-800 dark:text-blue-300">Total Portfolio Value</span>
                <span className="text-2xl font-bold text-blue-900 dark:text-blue-100">₹{totalValue.toLocaleString()}</span>
              </div>
            </div>

            <div className="space-y-4">
              {portfolio.map((row, i) => (
                <div key={i} className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Ticker</label>
                    <input
                      type="text"
                      value={row.ticker}
                      onChange={(e) => {
                        const updated = [...portfolio];
                        updated[i].ticker = e.target.value;
                        setPortfolio(updated);
                      }}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Amount (₹)</label>
                    <input
                      type="number"
                      value={row.amount}
                      onChange={(e) => {
                        const updated = [...portfolio];
                        updated[i].amount = parseFloat(e.target.value) || 0;
                        setPortfolio(updated);
                      }}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Asset Class</label>
                    <select
                      value={row.assetClass}
                      onChange={(e) => {
                        const updated = [...portfolio];
                        updated[i].assetClass = e.target.value;
                        setPortfolio(updated);
                      }}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-white"
                    >
                      <option>Stock</option>
                      <option>Bond</option>
                      <option>Gold</option>
                      <option>Cash</option>
                      <option>Real Estate</option>
                    </select>
                  </div>
                  <div className="flex items-end">
                    <button
                      onClick={() => {
                        const updated = portfolio.filter((_, index) => index !== i);
                        setPortfolio(updated);
                      }}
                      className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md font-medium transition-colors duration-200"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ))}
            </div>
            
            <button
              onClick={() => setPortfolio([...portfolio, { ticker: "", amount: 0, assetClass: "Stock" }])}
              className="mt-6 flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md font-medium transition-colors duration-200"
            >
              <Plus className="w-4 h-4" />
              Add Asset
            </button>
          </div>
        )}

        {page === "opt" && (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-6">
              <Settings className="w-6 h-6 text-blue-600" />
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Run Optimization</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">AI Model</label>
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-white"
                >
                  <option>Random Forest</option>
                  <option>XGBoost</option>
                  <option>LSTM</option>
                  <option>Transformer</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Strategy</label>
                <select
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-white"
                >
                  <option>MPT</option>
                  <option>Black-Litterman</option>
                  <option>Hybrid</option>
                  <option>Risk Parity</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Risk Level: {risk.toFixed(1)}
                </label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.1}
                  value={risk}
                  onChange={(e) => setRisk(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Conservative</span>
                  <span>Aggressive</span>
                </div>
              </div>
            </div>
            
            <button
              onClick={runOptimization}
              className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-md font-medium transition-colors duration-200 mb-6"
            >
              <Target className="w-4 h-4" />
              Optimize Portfolio
            </button>

            {optimized && (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-gray-700">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Ticker</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Current Amount</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Optimized Amount</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Change</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                    {optimized.map((item, i) => {
                      const change = ((item.optimizedAmount! - item.amount) / item.amount) * 100;
                      return (
                        <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">{item.ticker}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">₹{item.amount.toLocaleString()}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">₹{item.optimizedAmount!.toLocaleString()}</td>
                          <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {change >= 0 ? '+' : ''}{change.toFixed(1)}%
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {page === "viz" && (
          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center gap-3 mb-6">
                <PieChart className="w-6 h-6 text-blue-600" />
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Portfolio Visualizations</h2>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Portfolio Allocation</h3>
                  <div className="h-80">
                    <Pie
                      data={{
                        labels: portfolio.map((d) => d.ticker),
                        datasets: [
                          {
                            label: "Portfolio Allocation",
                            data: portfolio.map((d) => d.amount),
                            backgroundColor: [
                              "#FF6384",
                              "#36A2EB", 
                              "#FFCE56",
                              "#8BC34A",
                              "#9C27B0",
                              "#FF9800",
                              "#607D8B"
                            ],
                            borderWidth: 2,
                            borderColor: "#fff"
                          }
                        ]
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: {
                            position: 'bottom'
                          }
                        }
                      }}
                    />
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Investment Amount</h3>
                  <div className="h-80">
                    <Bar
                      data={{
                        labels: portfolio.map((d) => d.ticker),
                        datasets: [
                          {
                            label: "Investment Amount (₹)",
                            data: portfolio.map((d) => d.amount),
                            backgroundColor: "#42A5F5",
                            borderColor: "#1976D2",
                            borderWidth: 1
                          }
                        ]
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                          y: {
                            beginAtZero: true,
                            ticks: {
                              callback: function(value) {
                                return '₹' + value.toLocaleString();
                              }
                            }
                          }
                        }
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {page === "risk" && (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-6">
              <Brain className="w-6 h-6 text-blue-600" />
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Risk Profiling</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Risk Assessment</h3>
                <div className="space-y-4">
                  <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">Current Risk Level</h4>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${risk * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-medium text-blue-900 dark:text-blue-100">
                        {risk < 0.3 ? 'Conservative' : risk < 0.7 ? 'Moderate' : 'Aggressive'}
                      </span>
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="p-3 border border-gray-200 dark:border-gray-600 rounded-lg">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Volatility</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">{(risk * 20 + 5).toFixed(1)}%</span>
                      </div>
                    </div>
                    <div className="p-3 border border-gray-200 dark:border-gray-600 rounded-lg">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Expected Return</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">{(risk * 15 + 8).toFixed(1)}%</span>
                      </div>
                    </div>
                    <div className="p-3 border border-gray-200 dark:border-gray-600 rounded-lg">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">Sharpe Ratio</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">{(1.2 - risk * 0.3).toFixed(2)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Risk Recommendations</h3>
                <div className="space-y-3">
                  {risk < 0.3 && (
                    <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
                      <h4 className="font-medium text-green-900 dark:text-green-100 mb-2">Conservative Profile</h4>
                      <p className="text-sm text-green-800 dark:text-green-200">
                        Focus on bonds, fixed deposits, and blue-chip stocks. Consider increasing equity allocation gradually.
                      </p>
                    </div>
                  )}
                  {risk >= 0.3 && risk < 0.7 && (
                    <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
                      <h4 className="font-medium text-yellow-900 dark:text-yellow-100 mb-2">Moderate Profile</h4>
                      <p className="text-sm text-yellow-800 dark:text-yellow-200">
                        Balanced mix of equity and debt. Consider diversifying across sectors and asset classes.
                      </p>
                    </div>
                  )}
                  {risk >= 0.7 && (
                    <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                      <h4 className="font-medium text-red-900 dark:text-red-100 mb-2">Aggressive Profile</h4>
                      <p className="text-sm text-red-800 dark:text-red-200">
                        High equity allocation with growth stocks. Ensure adequate emergency fund and diversification.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {page === "bench" && (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3 mb-6">
              <Benchmark className="w-6 h-6 text-blue-600" />
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Benchmark Comparison</h2>
            </div>
            
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-700">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Asset</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Annual Return</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Risk (Volatility)</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Sharpe Ratio</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Max Drawdown</th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                  {[
                    { asset: "Your Portfolio", ret: 12.5, risk: 9.2, sr: 1.2, drawdown: -8.5, highlight: true },
                    { asset: "NIFTY 50", ret: 10.8, risk: 8.1, sr: 1.0, drawdown: -12.3, highlight: false },
                    { asset: "SENSEX", ret: 10.5, risk: 8.3, sr: 0.95, drawdown: -11.8, highlight: false },
                    { asset: "Gold ETF", ret: 7.2, risk: 4.1, sr: 0.9, drawdown: -6.2, highlight: false },
                    { asset: "Bank Nifty", ret: 13.2, risk: 12.5, sr: 0.85, drawdown: -18.7, highlight: false },
                    { asset: "IT Index", ret: 15.8, risk: 14.2, sr: 1.1, drawdown: -22.1, highlight: false }
                  ].map((row, i) => (
                    <tr key={i} className={`hover:bg-gray-50 dark:hover:bg-gray-700 ${row.highlight ? 'bg-blue-50 dark:bg-blue-900/20' : ''}`}>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${row.highlight ? 'text-blue-900 dark:text-blue-100' : 'text-gray-900 dark:text-gray-100'}`}>
                        {row.asset}
                        {row.highlight && <span className="ml-2 text-xs bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-200 px-2 py-1 rounded">You</span>}
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm ${row.ret > 10 ? 'text-green-600' : 'text-gray-900 dark:text-gray-100'}`}>
                        {row.ret}%
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm ${row.risk < 10 ? 'text-green-600' : 'text-orange-600'}`}>
                        {row.risk}%
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm ${row.sr > 1 ? 'text-green-600' : 'text-gray-900 dark:text-gray-100'}`}>
                        {row.sr}
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm ${row.drawdown > -10 ? 'text-green-600' : 'text-red-600'}`}>
                        {row.drawdown}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                <h4 className="font-medium text-green-900 dark:text-green-100 mb-1">Outperforming</h4>
                <p className="text-sm text-green-800 dark:text-green-200">Your portfolio beats NIFTY 50 by 1.7%</p>
              </div>
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-1">Risk-Adjusted</h4>
                <p className="text-sm text-blue-800 dark:text-blue-200">Sharpe ratio of 1.2 indicates good risk-adjusted returns</p>
              </div>
              <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                <h4 className="font-medium text-purple-900 dark:text-purple-100 mb-1">Diversification</h4>
                <p className="text-sm text-purple-800 dark:text-purple-200">Lower drawdown suggests better diversification</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;