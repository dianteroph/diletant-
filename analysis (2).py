import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import simplemma

from nltk import FreqDist, ngrams
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# pip install scipy==1.10.1
import gensim
from gensim import corpora
import spacy

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


stalin = pd.read_excel('stalin_output.xlsx', index_col=0)
stalin['person'] = 'stalin'
stalin['text'] = stalin['text'].fillna('')
stalin_filtered = stalin[stalin['text'].str.contains('сталин', case=False) &
                         ~stalin['text'].str.contains('сталинград', case=False)]

lenin = pd.read_excel('lenin_output.xlsx', index_col=0)
lenin['person'] = 'lenin'
lenin['text'] = lenin['text'].fillna('')
lenin_filtered = lenin[lenin['text'].str.contains('ленин', case=False) &
                              ~lenin['text'].str.contains('ленинград', case=False)]


merged = pd.concat([stalin_filtered, lenin_filtered]).reset_index(drop=True)

# Есть тексты, где обе фамилии встречаются, их мы тоже почистим
subset = merged[merged.duplicated(subset='link', keep=False) == True]
merged_unique = merged[merged.duplicated(subset='link') == False]

merged_unique['views'] = merged_unique['views'].str.replace('👀\n ', '', regex=False).str.replace('📅\n ', '', regex=False)
merged_unique['publication_date'] = merged_unique['publication_date'].str.replace('🖋\n ', '', regex=False).str.replace('📅\n ', '', regex=False)

merged_unique['author_cleaned'] = merged_unique['author'].str.replace('🖋\n ', '', regex=False).str.replace('📅\n ', '', regex=False)
merged_unique['publication_date'] = merged_unique.apply(lambda row: row['author_cleaned'] if row['publication_date'] == 'No 2nd element' else row['publication_date'], axis=1)
merged_unique.drop('author_cleaned', axis=1, inplace=True)
merged_unique.drop('author', axis=1, inplace=True)

merged_unique['text'] = merged_unique['text'].fillna('')

merged_unique['title'] = merged_unique['title'].str.lower()
merged_unique.rename(columns={'abstact':'abstract'}, inplace=True)
merged_unique['abstract'] = merged_unique['abstract'].str.lower()
merged_unique['text'] = merged_unique['text'].str.lower()


### TEXT PREPROCESSING

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    tokens = word_tokenize(text, language='russian')
    tokens = [token for token in tokens if token not in string.punctuation]
    special_punctuation = ['»', '«', '—', '-']
    tokens = [token for token in tokens if token not in special_punctuation]
    stop_words = set(stopwords.words('russian'))
    stop_words.update(['тыс', 'тыс.', 'о.', 'руб', 'руб.', 'в.', 'т', 'д', 'д.', 'г.', 'лет', 'гг.', 'этот', 'свой',
                       'из-за', 'это', 'который', 'такой'])
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatized_tokens = [simplemma.lemmatize(t, lang='ru') for t in filtered_tokens]
    lemmatized_tokens = [token for token in lemmatized_tokens if token not in stop_words]
    return lemmatized_tokens

merged_unique['processed_text'] = merged_unique['text'].apply(preprocess_text)

### PRELIMINARY ANALYSIS

# Frequency Distribution: Identifying the most common words or terms in the text.
all_tokens = [token for sublist in merged_unique['processed_text'] for token in sublist]
freq_dist = FreqDist(all_tokens)
plt.figure(figsize=(10, 5))
freq_dist.plot(30, title="Word Frequency Distribution")
plt.show()

# n-grams Analysis: Exploring common sequences of n words to understand phrases and context better.
bigrams = list(ngrams(all_tokens, 2))
bigram_freq = FreqDist(bigrams)
print("Most common bigrams:", bigram_freq.most_common(5))

# # Word Clouds: Visualizing the most frequent terms in a visually appealing format.
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(lemmatized_tokens))
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()

# Sentiment Analysis:
# Determining the sentiment or emotional tone behind the text, whether it is positive, negative, or neutral.
nlp = spacy.load('ru_core_news_sm')

tokenizer = AutoTokenizer.from_pretrained("sismetanin/rubert-ru-sentiment-rusentiment")
model = AutoModelForSequenceClassification.from_pretrained("sismetanin/rubert-ru-sentiment-rusentiment")

def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    print(outputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs).item()
    return sentiment

merged_unique['sentiment'] = merged_unique['text'].apply(get_sentiment)

merged_unique.sentiment.value_counts()

"""
1. Positive: This class indicates that the sentiment expressed in the text is positive or favorable.
2. Neutral: This class suggests that the sentiment expressed is neutral, without strong positive or negative feelings.
3. Negative: This class indicates that the sentiment expressed in the text is negative or unfavorable.
4. Speech Act: This class is used for texts that are performing an action rather than expressing sentiment (e.g., commands, questions).
5. Skip: This class is used for texts that do not fit into the other categories and should be skipped.
"""

# Topic Modeling:
# Discovering the abstract topics that occur in a collection of documents, using algorithms like Latent Dirichlet Allocation (LDA).

stalin_data = merged_unique[merged_unique['person'] == 'stalin']
lenin_data = merged_unique[merged_unique['person'] == 'lenin']

# Topic Modeling for Stalin
dictionary_stalin = corpora.Dictionary(stalin_data['processed_text'])
corpus_stalin = [dictionary_stalin.doc2bow(text) for text in stalin_data['processed_text']]
lda_model_stalin = gensim.models.LdaModel(corpus_stalin, num_topics=5, id2word=dictionary_stalin, passes=10)

print("Topics for Stalin:")
for idx, topic in lda_model_stalin.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# Topic Modeling for Lenin
dictionary_lenin = corpora.Dictionary(lenin_data['processed_text'])
corpus_lenin = [dictionary_lenin.doc2bow(text) for text in lenin_data['processed_text']]
lda_model_lenin = gensim.models.LdaModel(corpus_lenin, num_topics=5, id2word=dictionary_lenin, passes=10)

print("Topics for Lenin:")
for idx, topic in lda_model_lenin.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

