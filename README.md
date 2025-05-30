# price-prediction-from-news-sentiment
# Task 1: Exploratory Data Analysis (EDA) for News-Based Price Prediction

This task focuses on performing exploratory data analysis (EDA) on a dataset of news articles related to stock price prediction. The analysis includes descriptive statistics, time series analysis, text/topic modeling, and publisher analysis.

## Project Structure

```
price-prediction-from-news-sentiment/
│
├── data/
│   └── raw_analyst_ratings.csv
├── notebooks/
│   └── task.ipynb
├── src/
│   └── eda_analysis.py
└── README.md
```

## Main Files

- **notebooks/task.ipynb**: Jupyter notebook with step-by-step EDA, visualizations, and code explanations.
- **src/eda_analysis.py**: Python module containing reusable EDA functions for statistics, topic modeling, and publisher analysis.

## How to Use

1. **Install Requirements**
   - Make sure you have Python 3.8+.
   - Install dependencies:
     ```
     pip install pandas numpy matplotlib seaborn scikit-learn nltk
     ```

2. **Prepare Data**
   - Place your news dataset (e.g., `raw_analyst_ratings.csv`) in the `data/` folder.

3. **Run EDA Notebook**
   - Open `notebooks/task.ipynb` in Jupyter or VS Code.
   - Run the cells to see descriptive statistics, time series plots, and text/topic analysis.

4. **Use EDA Module**
   - Import and use functions from `src/eda_analysis.py` in your own scripts or notebooks:
     ```python
     from src.eda_analysis import load_data, text_length_statistics, extract_topics, analyze_time_series, analyze_publishers

     df = load_data('data/raw_analyst_ratings.csv')
     text_length_statistics(df)
     extract_topics(df, text_column='headline')
     analyze_time_series(df)
     analyze_publishers(df)
     ```

## Features

- **Descriptive Statistics**: Headline length, outlier detection, publisher frequency.
- **Time Series Analysis**: Articles per day, month, year, hour, and weekday.
- **Text Analysis**: Topic modeling using LDA.
- **Publisher Analysis**: Top publishers and domain extraction.

## Customization

- Change column names in function arguments if your dataset uses different names (e.g., `headline`, `content`, `publisher`, `date`).
- Adjust the number of topics or top words in `extract_topics`.

## Example Visualizations

- Distribution of headline lengths
- Publication frequency over time
- Articles published by hour of day
- Top publisher domains
- Top publications per selected stocks with historical data

