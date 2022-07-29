################################# 

# Karakter Dizisi Metotları (String Methods) 

################################# 

dir(str) 

# len 
dir(len) 

name = "tolga"
type(name) 
type(len) 

len("tolga") 

# eğer bir fonksiyon class yapısı içerisinde tanımlandıysa buna "method" denir 
# tanımlı değilse "fonksiyon" 

# upper() & lower() -> büyük küçük harf dönüşümleri 

"miuul".upper() 
"miUUl".lower() 


# replace : karakter değiştirir 

hi = "Hello Tolga"
hi.replace("l","p") 

# split : bölmek için 

"Hello Tolga".split() 

# strip : kırpar 
"ofofo".strip() 
"ofofo".strip("o")  

# capitalize : ilk harf büyür 

"foo".capitalize() 

dir("foo") 

"foo".startswith("f") 

