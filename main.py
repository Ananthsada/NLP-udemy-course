import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset/game_reviews.csv', encoding='latin1')
df = df[['user_review','user_suggestion']].dropna()
sns.factorplot(x='user_suggestion', data=df, kind='count', size=6, aspect=1.5, palette='gist_rainbow')
plt.show()