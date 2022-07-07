# uygulama : mülakat sorusu

# Amaç : çift sayıların karesini alarak bir sözlüğe eklemek istemektedir

# key'ler orjinal değerler value'lar ise değiştirilmiş değerler olacak

numbers = range(10)
new_dict = {}


for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2


# 2 yöntem mülakatta istenen

{n: n ** 2 for n in numbers if n % 2 == 0 }




