from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

### Part 1 - Frequency Analysis

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
column_names = ['sender', 'message', 'date']
df = pd.read_csv('../conversation_export.txt', names=column_names)
df['date'] = pd.to_datetime(df['date'])

df['message'] = df['message'].astype(str)
df = df[df['message'] != 'nan']  # Removing rows with 'nan' messages

all_messages = ' '.join(df['message'])
words = word_tokenize(all_messages)

filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

# Calculating the most common words and their frequencies
word_freq = Counter(filtered_words)
num_common_words = 20
common_words = [word for word, freq in word_freq.most_common(num_common_words)]
common_freqs = [freq for word, freq in word_freq.most_common(num_common_words)]

# Visualize frequencies using bar chart
plt.figure(figsize=(10, 6))
plt.barh(common_words, common_freqs, color='skyblue')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.title('Most Common Words in Messages')
plt.gca().invert_yaxis()  # Invert y-axis to have the most common words on top
plt.show()



### Part 2 - Sentiment Analysis

nltk.download('vader_lexicon')

column_names = ['sender', 'message', 'date']
df = pd.read_csv('../conversation_export.txt', names=column_names)
df['date'] = pd.to_datetime(df['date'])
sia = SentimentIntensityAnalyzer()

df['message'] = df['message'].astype(str)
df = df[df['message'] != 'nan'] 

df['sentiment_score'] = df['message'].apply(lambda message: sia.polarity_scores(message)['compound'])

# Classifying sentiments into 3 categories based on the compound score
def classify_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)

daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean()

# Visualizing the sentiment trend over time
plt.figure(figsize=(10, 6))
ax = daily_sentiment.plot(kind='line', color='blue')

num_ticks = 20
step = len(daily_sentiment) // (num_ticks - 1)
x_ticks = daily_sentiment.index[::step]
plt.xticks(x_ticks, rotation=45)

plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.title('Sentiment Trend Over Time')
plt.grid()
plt.tight_layout()
plt.show()
