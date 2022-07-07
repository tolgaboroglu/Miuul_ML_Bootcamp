### List Comprehension Uygulama 1  

import seaborn as sns 

df = sns.load_dataset("car_crashes") 
df.columns

for col in df.columns: 
    print(col.upper()) 

A = []

for col in df.columns:
    A.append(col.upper())

df.columns = A

