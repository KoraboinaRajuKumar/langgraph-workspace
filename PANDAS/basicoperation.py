import pandas as pd

df = pd.read_csv("dataset/career_choices_2026.csv")
#print(df["Preferred_Career_2026"])
print(df[["Preferred_Career_2026","Age"]])