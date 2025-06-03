import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import nltk
# Initialize NLTK's VADER sentiment analyzer (more suited for financial news)
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def align_data(news_df, stock_df):
    """
    Align news and stock data by date
    """
    # Convert dates to datetime format
    news_df['date'] = pd.to_datetime(news_df['date']).dt.date
    stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
    
    # Group news by date and calculate average sentiment
    daily_sentiment = news_df.groupby('date').apply(calculate_sentiment).reset_index()
    
    # Calculate daily stock returns
    stock_df['daily_return'] = stock_df['close'].pct_change() * 100
    
    # Merge the datasets
    merged_df = pd.merge(daily_sentiment, stock_df, on='date', how='inner')
    
    return merged_df.dropna()

def calculate_sentiment(news_group):
    """
    Calculate sentiment scores for a group of news articles
    """
    headlines = ' '.join(news_group['headline'].astype(str))
    
    # TextBlob sentiment
    tb_sentiment = TextBlob(headlines).sentiment
    
    # VADER sentiment
    vader_scores = sia.polarity_scores(headlines)
    
    return pd.Series({
        'textblob_polarity': tb_sentiment.polarity,
        'textblob_subjectivity': tb_sentiment.subjectivity,
        'vader_compound': vader_scores['compound'],
        'vader_positive': vader_scores['pos'],
        'vader_negative': vader_scores['neg'],
        'vader_neutral': vader_scores['neu'],
        'article_count': len(news_group)
    })

def analyze_correlation(merged_df):
    """
    Perform correlation analysis between sentiment and stock returns
    """
    # Calculate correlations
    correlations = {
        'textblob_polarity': pearsonr(merged_df['textblob_polarity'], merged_df['daily_return']),
        'vader_compound': pearsonr(merged_df['vader_compound'], merged_df['daily_return'])
    }
    
    # Print correlation results
    print("Correlation Analysis Results:")
    for metric, (corr, pval) in correlations.items():
        print(f"{metric}:")
        print(f"  Pearson r: {corr:.3f}")
        print(f"  p-value: {pval:.3f}")
        print("  " + interpret_correlation(corr, pval))
        print()
    
    # Visualize relationships
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.regplot(x='vader_compound', y='daily_return', data=merged_df)
    plt.title('VADER Compound Sentiment vs Daily Returns')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Daily Return (%)')
    
    plt.subplot(1, 2, 2)
    sns.regplot(x='textblob_polarity', y='daily_return', data=merged_df)
    plt.title('TextBlob Polarity vs Daily Returns')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Daily Return (%)')
    
    plt.tight_layout()
    plt.show()
    
    return correlations

def interpret_correlation(r, p):
    """Interpret correlation coefficient and significance"""
    strength = ""
    if abs(r) < 0.2:
        strength = "very weak"
    elif abs(r) < 0.4:
        strength = "weak"
    elif abs(r) < 0.6:
        strength = "moderate"
    elif abs(r) < 0.8:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if r > 0 else "negative"
    
    significance = "not statistically significant" if p > 0.05 else "statistically significant"
    
    return f"{strength} {direction} correlation ({significance})"

