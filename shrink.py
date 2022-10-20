import pandas as pd

df = pd.read_csv("Datasets/full.csv", header=None)
df1 = df[df.columns[0]]
df2 = df[df.columns[1000:4000]]
# df = df1.concat(df2, ignore_index = True)
# df = df.iloc[ : , [0, 1000:1002]]
df = pd.merge(df1, df2, right_index=True, left_index=True)
print (df)
df.to_csv(index=False, path_or_buf='Datasets/1000full5000.csv', header=False)
print("Done")
