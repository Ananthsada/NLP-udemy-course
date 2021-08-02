import pandas as pd
import spacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example
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
#nlp = spacy.blank("en")
#nlp.remove_pipe('tok2vec')
#nlp.remove_pipe('attribute_ruler')
#nlp.remove_pipe('lemmatizer')
textcat = nlp.add_pipe("textcat", last=True)
textcat.add_label("RECOMMENDED")
textcat.add_label("DONT_BUY")

df['tuples'] = df.apply(lambda row: (row['user_review'], row['user_suggestion']), axis=1)
train = df['tuples'].tolist()

(train_texts, train_cats), (dev_texts, dev_cats) = load_data(train, limit=13500, split=0.8)
train_data = list(zip(train_texts, [{'cats':cat} for cat in train_cats]))
n_iter = 10
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
print(other_pipes)
print(nlp.pipe_names)
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.create_optimizer()

    print("Training the model")

    for i in range(n_iter):
        print("iteration ", i)
        losses = {}
        batches = minibatch(train_data, size=compounding(4., 32., 1.001))
        for batch in batches:
            print("batch next")
            texts, annotations = zip(*batch)
            example = []
            # Update the model with iterating each text
            for i in range(len(texts)):
                doc = nlp.make_doc(texts[i])
                example.append(Example.from_dict(doc, annotations[i]))
            nlp.update(example, drop=0.2, losses=losses)