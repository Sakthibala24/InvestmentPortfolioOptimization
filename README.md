# 📊 Investment Portfolio Optimization using Machine Learning

This project applies machine learning algorithms to optimize asset allocation across **stocks, gold, and cash** using **5 years of Indian financial data**. It integrates predictive modeling and portfolio optimization techniques including **Modern Portfolio Theory (MPT)** and the **Black-Litterman Model** to generate optimized, risk-adjusted investment portfolios.

---

## 📌 Features

- ✅ Uses 5 years of historical Indian financial data
- ✅ Predictive models: XGBoost, LSTM, Random Forest
- ✅ Optimization models: MPT & Black-Litterman
- ✅ Risk profiling based on investor preferences
- ✅ Benchmark comparison with market indices
- ✅ Web app for interactive portfolio creation
- ✅ Export results to Excel

---

## 🧠 Methodology

1. **Data Collection**
   - Collected historical data for 100 Indian stocks, gold prices, and interest rates (cash) over 5 years.
   - Data saved in Excel format for efficient offline access.

2. **Data Preprocessing**
   - Cleaned and merged datasets.
   - Handled missing values, normalized features, and calculated returns.

3. **Feature Engineering**
   - Generated technical indicators and macroeconomic features.
   - Created lagged features for time-series prediction.

4. **Model Training**
   - Built and compared models (Random Forest, XGBoost, LSTM) to predict asset returns.
   - Evaluated performance using metrics like MSE, R², and Sharpe Ratio.

5. **Portfolio Optimization**
   - Used predictions to construct optimized portfolios via:
     - **Modern Portfolio Theory (MPT)**
     - **Black-Litterman Model**
   - Compared with equal-weighted and benchmark portfolios.

6. **Web Application**
   - Developed a user-friendly interface for:
     - Portfolio input
     - Risk profiling
     - Visualization of performance
     - Downloadable reports

---

## 🛠️ Tools & Frameworks Used

### 💻 Languages
- Python 3.x
- HTML/CSS (for web UI)

### 📚 Libraries
- **Data Processing**: `pandas`, `numpy`, `openpyxl`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Machine Learning**: `scikit-learn`, `xgboost`, `tensorflow/keras`
- **Time Series & Finance**: `yfinance`, `statsmodels`
- **Optimization**: `cvxpy`, `PyPortfolioOpt`, `pandas-datareader`
- **Web App**: `Flask` or `FastAPI`, `jinja2`, `wtforms`

### ⚙️ Tools
- Jupyter Notebook
- VS Code / PyCharm
- Git & GitHub
- Excel (for data storage and export)

---

## 🚀 Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/investment-portfolio-optimization-ml.git
cd investment-portfolio-optimization-ml
pip install -r requirements.txt
