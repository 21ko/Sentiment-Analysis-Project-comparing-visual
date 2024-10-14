import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import re
from collections import Counter

file_path = 'C:/Users/ya21/Downloads/proj2/essential/modified_data.csv'  # Adjust path
df = pd.read_csv(file_path, delimiter=';')


df.dropna(subset=['sentence', 'sentiment'], inplace=True)


def clean_text(text):
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'\W', ' ', text)  
    text = text.lower() 
    return text

df['clean_sentence'] = df['sentence'].apply(clean_text)


plt.figure(figsize=(8,5))
sentiment_counts = df['sentiment'].value_counts()
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['skyblue', 'lightgreen'])
plt.title('Sentiment Distribution (Matplotlib)', fontsize=16)
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(x='sentiment', data=df, palette='pastel')
plt.title('Sentiment Distribution (Seaborn)', fontsize=16)
plt.show()

fig = px.pie(df, names='sentiment', title='Sentiment Distribution (Plotly)', 
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()


all_text = ' '.join(df['clean_sentence'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off') 
plt.title('Word Cloud of All Text', fontsize=16)
plt.show()


for sentiment in df['sentiment'].unique():
    sentiment_text = ' '.join(df[df['sentiment'] == sentiment]['clean_sentence'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentiment_text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment.capitalize()} Sentiment', fontsize=16)
    plt.show()


df['sentence_length'] = df['clean_sentence'].apply(lambda x: len(x.split()))


plt.figure(figsize=(8,5))
sns.boxplot(x='sentiment', y='sentence_length', data=df, palette='pastel')
plt.title('Sentence Length Distribution by Sentiment (Seaborn)', fontsize=16)
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(df['sentence_length'], bins=20, kde=True, color='skyblue')
plt.title('Histogram of Sentence Lengths (Seaborn)', fontsize=16)
plt.xlabel('Number of Words', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()


fig = px.histogram(df, x='sentence_length', color='sentiment', nbins=20, title='Sentence Lengths by Sentiment (Plotly)')
fig.show()

word_list = ' '.join(df['clean_sentence']).split()
word_freq = Counter(word_list)


word_freq_df = pd.DataFrame(word_freq.items(), columns=['word', 'frequency']).sort_values(by='frequency', ascending=False)


plt.figure(figsize=(10,6))
sns.barplot(x='frequency', y='word', data=word_freq_df.head(20), palette='Blues_r')
plt.title('Top 20 Most Common Words', fontsize=16)
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Words', fontsize=12)
plt.show()