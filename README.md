# Price Prediction from News Sentiment

This project analyzes the relationship between news sentiment and stock price movements, and provides quantitative analysis of historical stock data using technical indicators.It also perform th6e correlation analysis

---

## Project Structure

```
price-prediction-from-news-sentiment/
│
├── data/
│   ├── raw_analyst_ratings.csv         # News headlines and metadata
│   └── finance_data/                   # Historical stock price CSVs
│        ├── AAPL_historical_data
│        ├── NVDA_historical_data
│        └── ... (other tickers)
├── notebooks/
│   ├── task.ipynb                      # News sentiment EDA and analysis
│   └── Quantitative analysis.ipynb     # Technical indicator and price analysis
├── scripts/
│   └── correlation.py                  # Functions for aligning and correlating news & price data
├── downloads/
│   └── ta_lib-0.6.3-cpXXX.whl          # TA-Lib wheel for Windows
└── README.md
```

---

## Features

### 1. News Sentiment Analysis (`notebooks/task.ipynb`)
- Loads and cleans news headline data.
- Calculates headline length statistics and visualizes distributions.
- Analyzes publication frequency by stock, publisher, and date.
- Extracts date components (year, month, day, hour, weekday) for time series analysis.
- Performs topic modeling on headlines using LDA.
- Calculates sentiment scores using TextBlob and VADER.
- Visualizes sentiment distributions and publication trends.

### 2. Quantitative Stock Analysis (`notebooks/Quantitative analysis.ipynb`)
- Loads historical stock price data for major tickers.
- Computes technical indicators with TA-Lib:
  - SMA, EMA, RSI, MACD
- Calculates daily and cumulative returns, and rolling volatility.
- Normalizes features for comparison and modeling.
- Visualizes price, indicators, returns, and volatility.

### 3. Correlation Analysis (`scripts/correlation.py`)
- Aligns news and stock data by date and ticker.
- Aggregates daily sentiment and merges with price data.
- Computes and visualizes correlation between sentiment and returns.

---

## How to Run

1. **Install Requirements**
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn nltk textblob
   pip install <path-to-ta-lib-wheel>
   ```

2. **Prepare Data**
   - Place news data in `data/raw_analyst_ratings.csv`.
   - Place historical stock CSVs in `data/finance_data/`.

3. **Run Notebooks**
   - Open `notebooks/task.ipynb` for news sentiment analysis.
   - Open `notebooks/Quantitative analysis.ipynb` for technical analysis.

4. **Use Correlation Functions**
   - Import from `scripts/correlation.py` in your notebook:
     ```python
     from scripts.correlation import align_data, analyze_correlation
     merged_df = align_data(news_df, stock_df)
     analyze_correlation(merged_df)
     ```
## Example Visualizations

- Distribution of headline lengths
- Publication counts by stock and publisher
- Sentiment distribution histograms
- Stock price with SMA/EMA overlays
- RSI and MACD indicator plots
- Correlation scatterplots between sentiment and returns

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, nltk, textblob
- TA-Lib (install via wheel for Windows)

