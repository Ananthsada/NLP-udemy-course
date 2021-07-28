import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer


# tokenization
nltk.download('punkt')

sentence="This is the sentence. I am going to tokenize it"
sentenceTokenized=sent_tokenize(sentence)
print(sentenceTokenized)

wordsTokenized=word_tokenize(sentence)
print(wordsTokenized)

print(sentence.split())
print(sentence.split('. '))

print(re.split('[. ]', sentence))


# Stemming test
Porter = PorterStemmer()
print(Porter.stem('Funny'))

Lancaster = LancasterStemmer()
print(Lancaster.stem('Funny'))

Snow = SnowballStemmer('english')
print(Snow.stem('Funny'))


# Lemmatization
nltk.download('wordnet')

lem = WordNetLemmatizer()

print(lem.lemmatize("kings"))