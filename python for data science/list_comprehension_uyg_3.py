
### List & Dictionary Comprehension Application 3

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

# Amaç Key'i string , value'su aşağıdaki gibi bir liste olan sözlük oluşturmak
# sadece sayısal değişkenler için yapmak istiyorum

#yöntem 1

num_cols = [col for col in df.columns if df[col].dtype !="0"]

soz = {}
agg_list = ["mean","min","max","sum"]

for col in num_cols :
    soz[col] = agg_list

# kısa yol
{col: agg_list for col in num_cols}

df[num_cols].head()
df.head()
df.tail()

## agg -> dicten alır cols a uygular içerisinde barındırır total: -> dict , num_cols-> ["mean"] birleştirir tabloyu


df[num_cols].agg(new_dict)
