import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

sentence="This is the sentence. I am going to tokenize it"
sentenceTokenized=sent_tokenize(sentence)
print(sentenceTokenized)

wordsTokenized=word_tokenize(sentence)
print(wordsTokenized)

print(sentence.split())
print(sentence.split('. '))

print(re.split('[. ]', sentence))