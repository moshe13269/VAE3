import pandas as pd

# p = "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0.csv"
# f = open(p)
# d = pd.read_csv(f)
# d1 = d.to_numpy()
# print(d[0])
# print(d[1])
# for index, row in d.iterrows():
#     print(row)
#     if index == 2:
#         break

c = 100000
tot = 0
i = 0
while c>0:
    i+=1
    c += c * (1.9 / 1200)
    c-= min(4250,c)
    tot += 4250
print('the pat for new money')
print(tot, i )
print('------------------------')

c1 = 56482
d0 = c1
tot = 0
i = 0
while c1>0:
    i+=1
    c1 += c1 * (3.38 / 1200)
    c1-= min(1188,c1)
    tot += 1188
print('the pat for 3.38')
print(tot, i )
print('can to save ', tot-d0)
print('------------------------')

c2 = 137943
d1 = c2
tot = 0
i = 0
while c2>0:
    i+=1
    c2 += c2 * (1.5 / 1200)
    c2-=  min(540,c2)
    tot += 540
print('the pat for 1.5', tot-d1)
print(tot, i )
print('------------------------')


c3 = 137943-45000
d2 = c3
tot = 0
i = 0
while c3>0:
    i+=1
    c3 += c3 * (1.5 / 1200)
    c3-=min(540,c3)
    tot += 540
print('the pat for 1.5 option 2', tot-d2)
print(tot, i )
print('------------------------')
