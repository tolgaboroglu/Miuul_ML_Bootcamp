############# 
# Set 
############# 

# - değiştirilebilir 
# - sırasız + eşittir 
# - kapsayıcıdır 

# difference() : iki kümenin farkı 

set1 = set([1,3,5,9])  
set2 = set([2,4,6,8])   

set1.difference(set2) 
set2.difference(set1)

# symetric_difference() : iki kümede de birbirlerine göre olmayanlar 

set1.symmetric_difference(set2) 

# iki kümenin kesişimi 

set1 = set([1,3,5,9])  
set2 = set([2,3,6,8]) 

set1.intersection(set2) 
set2.intersection(set1) 

# union() : iki kümenin birleşimi 

set.union(set1) 
set.union(set2) 

# iki kümenin kesişimi boş mu değil mi : isdisjoint()  

set1 = set([1,3,5,9])  
set2 = set([2,4,6,8]) 

set.isdisjoint(set2)  

# bir küme diğer kümenin alt kümesi mi 

set1.issubset(set2) 

#issuperset() : bir küme diğer kümeyi kapsıyor mu 

set2.issubset(set1) 


