######################

# Loops

######################

#for loop

students = ["Tolga","Melih","John","Maria"]

students[0]
students[1]


for i in students:
    print(i)

for i in students:
    print(i.upper())


salaries = [1000,222333,23456,21135]

for salary in salaries:
    print(salary)

# %20 ekstra

for salary in salaries:
    print(salary*20/100+salary)

# hem zam hem salary göstersin

def new_salary(salary,rate):
    return (int(salary*rate/100 + salary))

new_salary(15000,10)
new_salary(20000,20)

for salary in salaries:
    print(new_salary(salary,20))



