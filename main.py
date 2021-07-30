import pandas as pd
import re

def removeUserName(text):
    retText = re.sub('@[a-zA-z]+[a-zA-z0-9-_]', '', text)
    return retText

def removeHyperLink(text):
    retText = re.sub(r'http\S+', '', text)
    return retText

df = pd.read_csv('dataset/Coachella-2015-2-DFE.csv', encoding='latin1')
print(df['text'][1])
sampleText = removeUserName(df['text'][1])
print(sampleText)
sampleText = removeHyperLink(sampleText)
print(sampleText)