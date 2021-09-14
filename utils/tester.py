import pandas as pd

p = "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0.csv"
f = open(p)
d = pd.read_csv(f)
d1 = d.to_numpy()
print(d[0])
print(d[1])
# for index, row in d.iterrows():
#     print(row)
#     if index == 2:
#         break

