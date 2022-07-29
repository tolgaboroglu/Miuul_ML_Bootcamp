####################

#Enumerate : Otomatik / Indexer ile for loop

####################

students = ["John","Mark","Venessa","Mariam"]

for student in students:
    print(student)

for index, student in enumerate(students):
    print(index, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0 :
        A.append(student)
        print(index, student)
    else :
        B.append(student)
        