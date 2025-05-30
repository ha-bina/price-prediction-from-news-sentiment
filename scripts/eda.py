# EDA Module for News Articles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
from urllib.parse import urlparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

# Load your data
def load_data(file_path):
    return pd.read_csv(file_path)

# Descriptive Statistics
def text_length_statistics(df, text_column='headline', publisher_column='publisher', date_column='publication_date'):
    df['text_length'] = df[text_column].str.len()
    print("Basic text length statistics:")
    print(df['text_length'].describe())
    
    publisher_counts = df[publisher_column].value_counts()
    print("Top publishers:")
    print(publisher_counts.head(10))
    
    df[date_column] = pd.to_datetime(df[date_column])
    date_counts = df[date_column].dt.date.value_counts().sort_index()
    plt.figure(figsize=(12,6))
    date_counts.plot()
    plt.title('Publication Frequency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.show()

# Text Analysis (Topic Modeling)
def extract_topics(df, text_column='content', n_topics=5, n_top_words=10):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(df[text_column].dropna())
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    
    words = np.array(vectorizer.get_feature_names_out())
    for idx, topic in enumerate(lda.components_):
        print(f"Topic {idx+1}:")
        print(", ".join(words[topic.argsort()[:-n_top_words-1:-1]]))

# Time Series Analysis
def analyze_time_series(df, date_column='publication_date'):
    df[date_column] = pd.to_datetime(df[date_column])
    time_series = df[date_column].dt.hour.value_counts().sort_index()
    plt.figure(figsize=(10,5))
    sns.lineplot(x=time_series.index, y=time_series.values)
    plt.title("Articles Published by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Articles")
    plt.show()

# Publisher Analysis
def analyze_publishers(df, publisher_column='publisher'):
    domain_counts = Counter()
    for publisher in df[publisher_column].dropna():
        domain = publisher
        if '@' in publisher:
            domain = urlparse('http://' + publisher.split('@')[-1]).netloc
        domain_counts[domain] += 1
    print("Top Publisher Domains:")
    for domain, count in domain_counts.most_common(10):
        print(f"{domain}: {count}")



