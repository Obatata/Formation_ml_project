import pandas as pd
import numpy as np


df = pd.read_csv("heart_data.csv")
mask = np.random.rand(len(df)) < 0.8

train = df[mask]
test = df[~mask]


print('len tain : ', len(train))
print("len test : ", len(test))

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

