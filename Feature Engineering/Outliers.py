########################################

# FEATURE ENGINEERING & DATA PRE-PROCESSING

########################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#!pip install missingo
# conda install missingno

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.3f'%x)
pd.set_option('display.width',500)

def load_application_train():
    data = pd.read_csv('datasets/application_train.csv')
    return data

df = load_application_train()
df.head()



def load():
    data = pd.read_csv('datasets/titanic.csv')
    return  data

df = load()
df.head()

###########################

# 1.Outliers (Aykırı Değerler)

###########################

###########################

# Aykırı Değerleri Yakalama

###########################

# Grafik Teknikle Aykırı Değerler
# boxplot yanında sayısal değişken için "hist" grafiği de kullanılır.

sns.boxplot(x=df["Age"])
plt.show(block=True)

# Aykırı Değerler Nasıl Yakalanır

# Q1

q1 = df["Age"].quantile(0.25)
q1

# Q3
q3 = df["Age"].quantile(0.75)
q3

# IQR

iqr = q3 - q1
iqr

# üst sınır
up = q3 + 1.5* iqr
up

# alt sınır
low = q1 + 1.5 * iqr
low

#alt sınırdan büyük - üst sınırdan küçük olanları getirsin

df[(df["Age"] < low) | (df["Age"]> up)]

# index değerler

df[(df["Age"] < low) | (df["Age"]> up)].index

#Aykırı değer var mı yok mu ?/ bool

df[(df["Age"] < low) | (df["Age"]> up)].any(axis=None)

# aykırı olmayanları getir

df[~((df["Age"] < low) | (df["Age"]> up))]


df[(df["Age"] < low)].any(axis=None)

# Neler Yaptık ?
# 1.Eşik değer belirledik
# 2.Aykırılara eriştik
# 3.Hızlıca aykırı değer var mı yok mu diye sorduk

###########################################

#İşlemleri fonksiyonlaştırma :

###########################################

# eşit değerimizi tanımlayan fonksiyon

def outliers_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit , up_limit

outliers_thresholds(df,"Age")
outliers_thresholds(df,"Fare")

low, up = outliers_thresholds(df,"Fare")

df[(df["Fare"]< low) | (df["Fare"] > up).head()]

# indexlere erişmek istersek

df[(df["Fare"]< low) | (df["Fare"] > up).head()].index


# aykırı değerleri genellemek için/fonksiyonlaştırmak için

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outliers_thresholds(dataframe, col_name)
    # aslında yukarıada yaptığımız any yani bool ile herhangi bir boş aykırı değer var mı sorusuna denk gelir
    if dataframe[(dataframe[col_name]>up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df,"Age")
check_outlier(df,"Fare")



# Bunları tek tek mi yazacağız bir fonksiyona ihtiyacımız var ? / otomatik olarak seçecek


############################

# grap_col_names

############################


dff = load_application_train()
dff.head()

def grap_col_names(dataframe,cat_th=10, car_th=20):

    """
    Veri Setindeki kategorik,numerik ve kategorik fakat kordinal değişkenlerin isimlerini verir.
    Not : Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir

    Parameters
    ----------
           dataframe : dataframe
                 Değişken isimleri alınmak istenen dataframe
           cat_th : int,optional
                 Numerik fakat kategorik olan değişkenler için sınıf eşit değeri
           car_th : int , optional
                 Kategorik gibi gözüküp fakat kordinal değişkenler için sınıf eşik değeri


    Returns
    -------
         cat_cols: list
                 Kategorik değişken listesi
         num_cols: list
                 Numerik değişken listesi

    # bazı fonksiyonlar bool gibi ticket gibi kategorik ama sayısal olan değişkenleri de ifade edilmek için bu yöntemi uygulayacağız






    """


# subjectif yorum : kendimizce aralık belirliyoruz 10 dan büyükse 20 den küçükse bu projeden projeye değişir
# cat_cols, cat_but_car

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "0"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique()<cat_th and
            dataframe[col].dtypes !="0"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                dataframe[col].dtypes == "0"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]


# num_cols
    num_cols = [col for col in  dataframe.columns if dataframe[col].dtypes in ["int","float"] ]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations:{dataframe.shape[0]}")
    print(f"Variables:{dataframe.shape[1]}")
    print(f'cat_cols:"{len(cat_cols)}')
    print(f'num_cols:"{len(num_cols)}')
    print(f'cat_but_car:"{len(cat_but_car)}')
    print(f'num_but_cat:"{len(num_but_cat)}')
    return cat_cols,num_cols,cat_but_car

cat_cols,num_cols, cat_but_car = grab_col_names(df)

# variables = cat_cols + num_cols

# passengerId yi dışarda bırakalım veri setinin karışık olmaması için,bu yüzden veri setini baştan tanımlayalım

num_cols = [col for col in num_cols if col not in "PassengerId"]

num_cols

# outlier var mı ?/ aykırı değer

for col in num_cols:
    print(col ,check_outlier(df,col))

cat_cols, num_cols, cat_but_car = grab_col_names(df)



num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]
num_cols

for col in num_cols:
    print(col ,check_outlier(df,col))


##########################################################

# Aykırı Değerlere Programatik Olarak Ulaşmak
# Aykırı Değerlerin Kendilerine Erişmek

##########################################################


def grab_outliers(dataframe, col_name, index = False):
    low,up = outliers_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name]>low) | (dataframe[col_name] < up)].shape[0] > 10:
        print(dataframe[(dataframe[col_name]<low)| (dataframe[col_name] > up)].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name]> up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name]< low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df,"Age")

# indexini versin

grab_outliers(df,"Age",True)

# daha sonra kullanmak için saklamak istersek

age_index = grab_outliers(df,"Age",True)
age_index

# 3 şey yaptık
# 1- outlier_thresholds(df,"Age")  hesapladık
# 2- sadece bir değişkende "outlier" var mı yok mu ? check_outlier(df,"Age")
# 3- grab_outliers(df,"Age", True) yakaladık



##################################################

# Aykırı Değer Problemini Çözme

##################################################

#################

# Silme

#################
# aykırı değerleri tanımlamak için alt ve üst değerlere ihtiyacımız var

low , up = outliers_thresholds(df,"Fare")

# veri setinde kaç gözlem var ?
df.shape

# aykırı olmayanları getir

df[~((df["Fare"] < low)) | (df["Fare"] > up)].shape

# tüm ayrılıkları silmek için fonksiyon yazmak lazım
# alt sınır ve üst sınırın dışındakileri getir
def remove_outliers(dataframe,col_name):
    low_limit, up_limit = outliers_thresholds(dataframe,col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

cat_cols,num_cols,cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    new_df = remove_outliers(df,col)

df.shape[0] - new_df.shape[0]


######################################

# Baskılama Yöntemi (re-assignment with thresholds)

######################################

# veri kaybetmemek için bu yöntem kullanılabilir.Aykırı değerler yakalandıktan sonra
# eşik değerler ile değiştirilir.

low,up = outliers_thresholds(df,"Fare")

df[((df["Fare"]< low) | (df["Fare"] > up))]["Fare"]

# üstteki yöntemin loc ile yapımı

df.loc[((df["Fare"]< low) | (df["Fare"] > up)),"Fare"]


# üst sınıra göre seçme ve atama işlemi yapalım

df.loc[(df["Fare"] > up),"Fare"]

# aykırıların yerine up olarak atama

df.loc[(df["Fare"] > up),"Fare"] = up

# altı da deneyelim gelmese de

df.loc[(df["Fare"] > low),"Fare"] = low


# replace_with_thresholds
# alt limit altındakini = alt limitle
# üst limit yukarısındakini = üst limitle eşitle

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outliers_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable]<low_limit),variable] = low_limit
    dataframe.loc[(dataframe[variable]>up_limit),variable] = up_limit

df = load()

cat_cols , num_cols,cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

# outlier var mı yok mu ?

for col in num_cols:
    print(col,check_outlier(df,col))

for col in num_cols:
    replace_with_thresholds(df,col)

for col in num_cols:
    print(col,check_outlier(df,col))




####################################

# RECAP

####################################
#replace,thresholds,grab_col_name önem sırası

df = load()

# aykırı değer saptandı
outliers_thresholds(df,"Age")

# bu thresholdsa göre outlier var mı yok mu ?
check_outlier(df,"Age")

# outlierları bize getir
grab_outliers(df,"Age",index=True)

# tedavi edelim 891 di sildik 775
remove_outliers(df,"Age").shape

# baskıla/ değiştir
replace_with_thresholds(df,"Age")


##############################################

# Çok Değişkenli Aykırı Değer Analizi : Local Outlier Factor

##############################################

# 1. değişken yaş 2.evlilik sayısı
# 3 kere evlenmek aykırı değer değildir
# 17 de yaş için anormal değildir
# 17 yaşında olup 3 defa evlenmek aykırı değerdir
# tek başına değerlendirdiğinde aykırı olamayacak değerler birlikte ele alındığında aykırı olabilir.

# LOF : gözlemleri bulundukları konumdaki yoğunluk tabanlı skorlayarak buna  göre aykırı değer tanımı yapmamızı sağlar
# bir noktanın local yoğunluğu demek etrafındaki komşulukları demektir . Eğer bir nokta komşularının yoğunluğundan anlamılı
# şekilde düşük ise bu durumda bu nokta daha seyrek bölgededir yani bu aykırı değer olabilir.
# uzaklık skoru hesaplamamızı sağlar. 1 e ne kadar yakınsa o kadar iyi . 1 den uzaklaştıkça ilgili gözlemin
# outlier olma olasılığı artar
# tresholds aralık değerini ben belirlersem hangisi aykırı hangisi değil seçebilirim.

# elimizde 100 değişken var 2 boyutta görselleştir ?
# 2 bileşene  indirgeriz / PCA yöntemi ile


df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float', 'int64'])
df = df.dropna()
df.head()

# check_outlier
for col in df.columns:
    print(col,check_outlier(df,col))

low,up = outliers_thresholds(df,"carat")

# carat içerisinde kaç tane aykırı gözlem var

df[((df["carat"] < low) | (df["carat"] > up))].shape

# depth içerisinde kaç aykırı gözlem var

df[((df["depth"] < low) | (df["depth"] > up))].shape

# ağaç yöntemi kullanırsanız verileri silmemeniz gerekir

# çok değişkenli yaklaşalım

clf = LocalOutlierFactor(n_neighbors=20)

# üsttekini veri setimize uygulayalım

clf.fit_predict(df)

# takip edilebilirlik açısından

df_scores = clf.negative_outlier_factor_
df_scores[0:5]

# - değerlerle değerlendirmek istemezsek

-df_scores[0:5]

# fakat - değerlerle değerlendireceğiz eşit değerini daha net okuyabiliriz .

# - li değerler olduğu için -1 e yakın olması inline olma durumunu yükseltir

# k den b ye sırala

np.sort(df_scores)[0:5]

# eşik değer belirleme noktası
# Pca de kullanılan dirsek yöntemi
# eşik değerini bulmak için yapacaz

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked = True, xlim=[0,20],style = '.-')
plt.show(block = True)

# eğimin en dik olduğu değişimin ilk gözlemlendiği yere bakacaksın -> eşik değer
# bir yorum yapıyorum ve 3. değeri seçiyorum

th = np.sort(df_scores)[3]
th

# üstekinden daha - değer olanları aykırı değer olarak belirleyeceğim

df[df_scores < th]

df[df_scores < th].shape

# bunlar acaba neden aykırı ?

df.describe([0.01,0.05,0.75,0.90,0.99]).T

# mesala 79 almış bir değeri fakat karşılığını denkleştirecek bir değer yok çoklu değişkeni sağlyacak o yüzden
# 78.2 alıyor çünkü price ile ilişkili

df[df_scores < th].index

# silebiliriz aykırıları

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# baskılayabiliriz -> aykırılığın yerine başka gözlem koymamız gerek / eğer çok gözlem sayısı olursa bozulmalar olabilir(deneme)
# agaç yöntemi ise hiç dokunma ya da trasholders kullan -> aykırılara dokunma
# gözlem sayısı az ise baskılama kullanabilirsin
# doğrusal yöntemler kullanıyorsak aykırı yöntemler tüm ciddiyetini koruyor demektir

