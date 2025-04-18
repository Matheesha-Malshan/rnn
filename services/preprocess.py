import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df=pd.read_csv(r"C:\Users\HP\rnn\rnn\services\NFLX.csv")

df['Date']=pd.to_datetime(df['Date'])

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['day_of_week'] = df['Date'].dt.dayofweek
df['week_of_year'] = df['Date'].dt.isocalendar().week

df = df.drop('Date', axis=1)
df['week_of_year']=df['week_of_year'].astype('int32')
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix Heatmap")

df.drop(['Adj Close', 'day_of_week'],axis=1,inplace=True)

target=df['Close']
data=df.drop('Close',axis=1)

print(df)
