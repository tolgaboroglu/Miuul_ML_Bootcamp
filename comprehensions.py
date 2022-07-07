#############
# COMPREHENSIONS
############

##############

# List Comprehension -> tek bir satırda işlem yaparak işimizi kolaylaştırırız.

##############

salaries = [1000,2000,3000,4000,5000]
[salary *2 for salary in salaries]

[salary * 2 for salary in salaries if salary < 3000]

# else ile kullanımı

[salary * 2 if salary <3000 else salary * 0 for salary in salaries]

# zam işlemi
[new_salary(salary*2)if salary <3000 else new_salary(salary * 0.2) for salary in salaries]

# iki liste var bir listede istemediğim öğrenciler var istemediğim öğrencileri büyük harfle yazma

students = ["John","Mark","Venessa","Mariam"]

students_no = ["John","venessa"]

[student.lower() if student in students_no else student.upper() for student in students]

# dict comprehension

dictionary = { 'a': 1,
               'b': 2,
               'c': 3,
               'd': 4,

}

dictionary.keys()
dictionary.values()
dictionary.items()

# her bir value nin karesini almak istersek

{k: v ** 2 for (k,v) in dictionary.items()}

# keylere işlem yapmak istersek

{k.upper(): v for (k,v) in dictionary.items()}

# aynı anda hem key hem value değiştirme

{k.upper(): v*2 for (k,v) in dictionary.items()}


