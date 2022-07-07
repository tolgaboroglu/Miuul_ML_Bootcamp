# görev 1
x = 8
type(x)
y = 3.2
type(y)
z = 8j + 18
type(z)
a = "Hello World"
type(a)
b = True
type(b)
c = 23 < 22
type(c)
l = [1, 2, 3, 4]
type(l)
d = {"Name": "Jake",
     "Age": 27,
     "Address": "Downtown"}
type(d)
t = {"Machine Learning", "Data Science"}
type(t)
s = {"Python", "Machine Learning", "Data Science"}
type(s) 

# Görev 2

text = "The goal is to turn data into information, and information into insight."
text = text.upper()
text = text.replace(",", " ")
text = text.replace(".", " ")
text = text.split()
print(text) 

# Görev 3

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]
len(lst)
lst[0]
lst[10]

new_lst = lst[0:4]
lst.pop(8)
lst.append("A")
lst.insert(8, "N") 

# Görev 4 

dict = {'Christian': ["America", 18],
        "Daisy": ["England", 12],
        "Antonio": ["Spain", 22],
        "Dante": ["Italy", 25]}
dict.keys()
dict.values()
dict["Daisy"][1] = 13
dict["Ahmet"] = ["Turkey", 24]
dict.pop("Antonio") 

# Görev 5

l = [2, 13, 18, 93, 22]
even_list = []
odd_list = []

def func(l):
    even_local = []
    odd_local = []
    for num in l:
        if num % 2 == 0:
            even_local.append(num)
        else:
            odd_local.append(num)
    return even_local, odd_local 

even_list, odd_list = func(l) 

# Görev 6 

#all vars .upper(), numeric vars NUM_ + .upper()
import seaborn as sns
df = sns.load_dataset("car_crashes")

df.columns = ["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns] 

# Görev 7 

# if NO does not exist: .upper() + "_FLAG", if exists: .upper()
import seaborn as sns
df = sns.load_dataset("car_crashes")

b = [col.upper() if "no" in col else col.upper() + "_FLAG" for col in df.columns] 

# Görev 8 

import seaborn as sns
df = sns.load_dataset("car_crashes")
og_list = ["abbrev", "no_previous"]
df.columns

new_cols = [col for col in df.columns if col not in og_list]
new_df = df[new_cols]