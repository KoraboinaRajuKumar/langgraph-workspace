import pandas as pd

df = pd.read_csv("dataset/career_choices_2026.csv")
#print(df.head())  # first 5 
# print(df.tail()) last 5
#print(df.shape)  #(200, 7)
#print(df.columns) # list of columns
print(df.info())

 

