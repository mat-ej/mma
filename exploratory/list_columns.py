import pandas as pd
df = pd.read_csv('/home/m/repo/mma/products/data/data.csv', parse_dates=['DATE'])
col_list = df.columns.to_list()

print('[')
for col in col_list:
    print("'" + col + "',")
print(']')
# print(df)