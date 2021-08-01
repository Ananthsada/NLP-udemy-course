import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import random

def load_data(data, limit = 0, split = 0.8):
    train_data = data
    random.shuffle(train_data)
    text, labels = zip(*train_data)
    cats = [{'RECOMMENDED': bool(y), 'DONT_BUY': not bool(y)} for y in labels]
    split = int(len(text) * split)
    return (text[:split], cats[:split]), (text[split:], cats[split:])

df = pd.read_csv('dataset/game_reviews.csv', encoding='latin1')
df = df[['user_review','user_suggestion']].dropna()
#sns.factorplot(x='user_suggestion', data=df, kind='count', size=6, aspect=1.5, palette='gist_rainbow')
#plt.show()

nlp = spacy.load("en_core_web_sm")
textcat = nlp.add_pipe("textcat", last=True)
textcat.add_label("RECOMMENDED")
textcat.add_label("DONT_BUY")

df['tuples'] = df.apply(lambda row: (row['user_review'], row['user_suggestion']), axis=1)
train = df['tuples'].tolist()

(train_texts, train_cats), (dev_texts, dev_cats) = load_data(train, limit=13500, split=0.8)
train_data = list(zip(train_texts, [{'cats':cat} for cat in train_cats]))
print(train_data[:2])