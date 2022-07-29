##########################
# Uygulama - Mülakat Sorusu

##########################

# Amaç : aşağıdaki şekilde string değiştiren fonksiyon yazmak istiyoruz

# before : "hi my name is john and i am learning python"
# after : "Hi mY Name iS JoHn aNd i aM LearNiNg PYtHoN"

range(len("miuul" ))
range(0,5)

for i in range(0,5):
    print(i)

4 % 2 == 0



def alternating(string):
    new_string = ""
    # girilen string'in index'lerinde gez
    for string_index in range(len(string)):
        # index çift ise büyük harfe çevir
        if string_index % 2 == 0:
            string[string_index].upper()
            new_string += string[string_index].upper()
            #index tek ise küçük harfe çevir
        else:
            new_string += string[string_index].lower()
    print(new_string)

alternating("miuul")



