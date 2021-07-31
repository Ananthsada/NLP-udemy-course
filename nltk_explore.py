import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')

def getHashtags(text):
    retText = re.findall('#(\w+)', text)
    return retText

def removeHastags(text):
    retText = re.sub('#(\w+)', '', text)
    return retText

def removeUserName(text):
    retText = re.sub('@[a-zA-z]+[a-zA-z0-9-_]', '', text)
    return retText

def removeHyperLink(text):
    retText = re.sub(r'http\S+', '', text)
    return retText

def removeNonAscii(text):
    retText = ''.join(i for i in text if ord(i) < 128)
    return retText

def removeStopWords(text):
    stops = set(stopwords.words('english'))
    stops.update('mailto', 'this', 'it')
    retText = ' '.join(i for i in text.split() if i not in stops)
    return retText

def convetLowerCase(text):
    retText = text.lower()
    return retText

def removeEmailAddress(text):
    retText = re.sub('[\w\.-]+@[\w]+.[\w]+', '', text)
    return retText

def removePunctuation(text):
    retText = re.findall('\w+', text)
    return ' '.join(each for each in retText)

def removeSpecialCharacters(text):
    retText = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return retText

def explore_nltk():
    df = pd.read_csv('dataset/Coachella-2015-2-DFE.csv', encoding='latin1')
    df['hastag'] = df.text.apply(func = getHashtags)
    df['new_tweet'] = df.text.apply(func = removeUserName)
    df['new_tweet'] = df.new_tweet.apply(func = removeHyperLink)
    df['new_tweet'] = df.new_tweet.apply(func = removeNonAscii)
    df['new_tweet'] = df.new_tweet.apply(func = convetLowerCase)
    df['new_tweet'] = df.new_tweet.apply(func = removeEmailAddress)
    df['new_tweet'] = df.new_tweet.apply(func = removeStopWords)
    df['new_tweet'] = df.new_tweet.apply(func = removeHastags)
    df['new_tweet'] = df.new_tweet.apply(func = removePunctuation)
    df['new_tweet'] = df.new_tweet.apply(func = removeSpecialCharacters)
    for i in range(0, 5, 1):
        print(df['text'][i])
        print(df['new_tweet'][i])