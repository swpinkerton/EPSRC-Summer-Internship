import pandas as pd
df = pd.read_csv("Datasets/full.csv", header=None)
df = df.std(axis=1)
print (df.min(axis=0))