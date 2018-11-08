import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.read_csv('E:\Database\label.csv')
userID = df['user.id']
imageNames = df['image']
emotionLabels = df['emotion']

print(len(userID))

