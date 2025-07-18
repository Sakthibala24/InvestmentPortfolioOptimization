import yfinance as yf
import pandas as pd
from datetime import datetime

# Hardcoded list of 100 NSE stock tickers (you can adjust this list)
tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "LT.NS",
    "SBIN.NS", "KOTAKBANK.NS", "ITC.NS", "AXISBANK.NS", "HINDUNILVR.NS", "ASIANPAINT.NS",
    "BAJFINANCE.NS", "MARUTI.NS", "HCLTECH.NS", "ULTRACEMCO.NS", "NTPC.NS", "SUNPHARMA.NS",
    "TECHM.NS", "POWERGRID.NS", "NESTLEIND.NS", "TITAN.NS", "COALINDIA.NS", "WIPRO.NS",
    "TATAMOTORS.NS", "JSWSTEEL.NS", "DRREDDY.NS", "CIPLA.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "M&M.NS", "BHARTIARTL.NS", "GRASIM.NS", "BPCL.NS", "IOC.NS", "BRITANNIA.NS",
    "DIVISLAB.NS", "HINDALCO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "SHREECEM.NS",
    "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS", "SBILIFE.NS", "ICICIPRULI.NS",
    "HDFCLIFE.NS", "UPL.NS", "TATACONSUM.NS", "VEDL.NS", "GAIL.NS", "AMBUJACEM.NS",
    "DMART.NS", "DABUR.NS", "HAVELLS.NS", "SRF.NS", "PIDILITIND.NS", "LUPIN.NS",
    "BOSCHLTD.NS", "TORNTPHARM.NS", "MUTHOOTFIN.NS", "ABB.NS", "NAVINFLUOR.NS",
    "PEL.NS", "PAGEIND.NS", "GODREJCP.NS", "INDIGO.NS", "AUROPHARMA.NS", "ZOMATO.NS",
    "IRCTC.NS", "NYKAA.NS", "GLAND.NS", "CHOLAFIN.NS", "BEL.NS", "ATGL.NS", "POLYCAB.NS",
    "ICICIGI.NS", "IDFCFIRSTB.NS", "BANKBARODA.NS", "TVSMOTOR.NS", "LICI.NS",
    "HAL.NS", "RECLTD.NS", "BAJAJHLDNG.NS", "NHPC.NS", "CANBK.NS", "BANDHANBNK.NS",
    "BHEL.NS", "UNIONBANK.NS", "PNB.NS", "IDBI.NS", "IOB.NS", "MRF.NS", "SJVN.NS",
    "HFCL.NS", "RVNL.NS", "NATIONALUM.NS", "JINDALSTEL.NS", "TRIDENT.NS"
]

start_date = "2019-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

all_data = []

for ticker in tickers:
    print(f"📥 Fetching {ticker}...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        df["Ticker"] = ticker
        df.reset_index(inplace=True)  # Ensures 'Date' is a column
        all_data.append(df)
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")

# Combine all data and save to CSV
combined_df = pd.concat(all_data, ignore_index=True)
combined_df.to_csv("stock_data.csv", index=False)
print("✅ Saved final stock_data.csv with Date column.")
