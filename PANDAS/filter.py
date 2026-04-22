import pandas as pd

df = pd.read_csv("dataset/career_choices_2026.csv")

# Filter rows where Age is greater than 25
#df_filtered = df[df['Age']>=25]
#print(df_filtered)
print(df[df["Interested_in_Generative_AI"]=="Yes"])



