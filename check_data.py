import pandas as pd
import numpy as np
import random
"link dataset : https://www.kaggle.com/fedesoriano/heart-failure-prediction/version/1"
df = pd.read_csv("train.csv")

print(len(df))
print(df.columns)




percentage_cholesterol = round(len(df)*20/100)
percentage_chestPainType = round(len(df)*5/100)
percentage_maxHR = round(len(df)*0.5/100)
percentage_st_slop = round(len(df)*25/100)


list_index = list(range(len(df)))
print("---")
print(list_index)
print("cholesterol")
random.shuffle(list_index)
index_cholesterol = list_index[0:percentage_cholesterol]
df.loc[index_cholesterol, "Cholesterol"] = ''
print(index_cholesterol)
print("------")
print("chestPainType")
random.shuffle(list_index)
index_chestPainType = list_index[0:percentage_chestPainType]
df.loc[index_chestPainType, "ChestPainType"] = ''
print(index_chestPainType)
print("------")
print("maxHR")
random.shuffle(list_index)
index_maxHR = list_index[0:percentage_maxHR]
df.loc[index_maxHR, "MaxHR"] = ''
print(index_maxHR)
print("------")
print("st_slop")
random.shuffle(list_index)
index_ST_Slope = list_index[0:percentage_st_slop]
df.loc[index_ST_Slope, "ST_Slope"] = ''
print(index_ST_Slope)
print("------")
df.to_csv("heart_data.csv", index=False)
