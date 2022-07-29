###########################################

# Missing Values

###########################################

# Ne demek ?
# Gözlemlerde eksik olması durumunu ifade etmektedir.

# Eksik veri problemi nasıl çözülür ?
# 1.silme
# 2.değer atama yöntemleri -> ortalama,median
# 3.tahmine dayalı yöntemler -> istatiksel


# Not :
# eksik veri ile çalışırken göz önünde bulundurulması
# gereken önemli konulardan birisi : Eksik verinin rassallığı(rastgele ortaya çıkıp çıkmadığı)

#Not2 :
# eksik değerlere sahip gözlemlerin veri setinden direkt çıkarılması ve rassallığının incelenmemesi,
# yapılacak istatiksel çıkarımların ve modelleme çalışmalarının güvenilirliğini düşürecektir

#Not3 :
# eksik gözlemlerin veri setinden direkt çıkarılabilmesi için veri setindeki eksikliğin bazı durumlarda kısmen
# bazı durumlarda tamamen rastlantısal olarak oluşmuş olması gerekmektedir.
# Eğer eksiklikler değişkenler ile ilişkili olarak ortaya çıka yapısal problemler ile meydana gelmiş ise bu durumda
# yapılacak silme işlemleri ciddi yanlılıklara sebep olabilecektir.


# kısaca NA ler rastgele çıktıysa sorun yok ister sileriz , ister atarız
# NA = 0
# başka değerlerle ilişkisine bakılır değiştirmeden önce / rastgele değilse dikkatli ol


###############################################

# Eksik Değerleri Yakalama

###############################################

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

def load():
    data = pd.read_csv('datasets/titanic.csv')
    return  data

df = load()
df.head()

# verilerin tamamında eksik değer var mı yok mu ? hızlı sorusunu sor

df.isnull().values.any()

# değişkenlerdeki eksik değer sayısı

df.isnull().sum()

# true 1 false 0

# değişkenlerdeki tam değer sayısı

df.isnull().sum()

# veri setindeki toplam eksik değer sayısı

df.isnull().sum().sum()

# en az bir tane eksik değere sahip olan gözlem birimleri

df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri

df[df.notnull().all(axis=1)]

#indexlerine eriş

df[df.isnull().any(axis=1)].index

# azalan şekilde sıralamak

df.isnull().sum().sort_values(ascending = False)

# eksikliğin veri setindeki oranı

(df.isnull().sum() / df.shape[0]*100).sort_values(ascending=False)

# sadece eksik değere sahip olan değişkenleri seçme yöntemi

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0 ]
na_cols

# bunların hepsini tek fonksiyonda yaz

def missing_values_table(dataframe,na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss,np.round(ratio,2)], axis=1, keys=['n_miss','ratio'])
    print(missing_df,end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

# eksik değerlerin isimleri

missing_values_table(df,True)


########################################

# Eksik Değer Problemini Çözme

########################################

# eğer ağaca dayalı yöntem kullanılırsa eksik değerler göz ardı edilebilir
# bağımlı değişkenlerden dolayı işlemler uzayabilir

#######################

# Çözüm 1: Hızlıca Silmek

#######################

df.dropna().shape

######################

# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak

######################

df["Age"].fillna(df["Age"].mean())


df["Age"].fillna(df["Age"].mean()).isnull().sum()

# sabit bir değerle doldurma/eksik değer yerine 0 yazmak

df["Age"].fillna(0).isnull().sum()

# birçok işlem olursa napacağım ?
# apply ve lambda kullanılabilir

df.apply(lambda x:x.fillna(x.mean()), axis=0)

# öyle bir işlem yapmalıyım ki sadece sayısal değişkenlerle doldursun
# bu işlem eğer objecten farklı ise 0 ile doldur
# farklı değilse olduğu gibi kalsın


df.apply(lambda x:x.fillna(x.mean) if x.dtype != "0" else x , axis=0).head()

dff = df.apply(lambda x:x.fillna(x.mean) if x.dtype != "0" else x , axis=0).head()
dff

dff.isnull().sum().sort_values(ascending=False)

#Not : kategorik değişkenler için en mantıklı doldurma yöntemlerden birisi "mod"unu almak

df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()


# eğer dilerseniz herhangi ifade kullanabilirsiniz
# daha sonra analiz edip durum değerlendirmesi yapacaksınız

df["Embarked"].fillna("missing")

# otomatik olarak yapma

df.apply(lambda x:x.fillna(x.mode()[0]) if(x.dtype == "0" and len(x.unique()) <= 10)else x, axis=0).isnull().sum()



#############################

# Kategorik Değişken Kırılımında Değer Atama

#############################

# veri setindeki bazı kategorik değişkenleri ele almak ve bunları veri setinde

df.groupby("Sex")["Age"].mean()

# sadece yaşın ort
df["Age"].mean()

# her iki cinsiyetinde yaş ortalaması farklı olduğundan cinsiyete göre değer atamak daha doğru

# programatik şekilde yapımı
# erkekleri ve kadınları ayrı ayrı alacağız
# kadın ve erkeğe göre groupby alıp yaş ortalamasını hesaplamalıyım
# yaş ortalamasını da cinsiyet kırılımına göre uyarlayalım

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

# daha açık şekilde yapalım

# yaş değişkenin de eksiklik olup cinsiyeti kadın olanı getirdi

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female")]

# kadınlara yönelik groupby

df.groupby("Sex")["Age"].mean()["female"]

# yukardaki ikisin birleştiriyoruz

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"),"Age"] = df.groupby("Sex")["Age"].mean()["female"]


# erkekler için de aynı işlemi yapalım
df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"),"Age"] = df.groupby("Sex")["Age"].mean()["male"]

# eksik değer var mı bakalım

df.isnull().sum()

####################################################

# Çözüm 3: Tahmine Dayalı Atama İle Doldurma

####################################################

# gelişmiş yöntemlerle eksiklikleri doldurmak

# modelleme işlemi yapacağız
# onehub encoder
# bir standard oluşturacağız ona uymak lazım (KNN)

df = load()

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



cat_cols,num_cols,cat_but_car  = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

num_cols

# cat_cols lara dönüşüm işlemi yapmam gerek encoder olması lazım = label encoding,onehub encoding
# onehut
#binary şekilde gösterecek /numeric
# sadece kategorik değişkenleri çevirir
# 3 olanları dokunmuyoruz şimdilik

dff = pd.get_dummies(df[cat_cols + num_cols],drop_first=True)

dff.head()

# değişkenleri standartlaştırma

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff),columns=dff.columns)

dff.head()

# KNN'in uygulanması/eksik değerleri doldurmak için
# KNN : bana arkadaşını söyle sana kim olduğunu söylim
# yaş değişkenindeki eksik değerlerindeki en yakın 5 komşusunu baz alacak
# en yakın 5 komşunun yaş ortalamasını alacak

from sklearn.impute import KNNImputer

# model nesnesi oluşturulur
imputer  = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

# doldurduğum yerleri görmek istiyorum
# gerçek değer değil standartlaştırılmış değerler
# standartlaştırılmış değerleri geri alacam
# scaler.inverse_transform(dff)

dff = pd.DataFrame(scaler.inverse_transform(dff),columns=dff.columns)
dff.head()

# kıyaslamam lazım

df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(),["Age","age_imputed_knn"]]

#detaylı incelemek için

df.loc[df["Age"].isnull()]


#######################

#RECAP

#######################

# 1. veriyi yükledik
# 2. eksik veriyi raporladık
# 3. sayısal değişkenleri ortalama veya median ile değiştirdik
# 4. kategorik değişkenleri modu ile doldurduk
# 5. kategorik değişken kırılımında sayısal değişkenleri doldurmak
# 6. tahmine dayalı atama ile doldurma




##############################################

# Gelişmiş Analizler

##############################################
# bar yöntemi tam sayıları vermektedir

msno.bar(df)
plt.show(block=True)

# matrix methodu

# görseldek eksiklikliklerin bir araya çıkma durumunu görsel bir araçtır
# mesala age değişkenindeki siyah veya beyaz yapıların başka bir değişken olan sex değişkeninde de aynı şekilde noktaların,
# boşlukların olmasını bekleriz.

msno.matrix(df)
plt.show(block=True)

# heatmap
# ısı haritası
# eksik değerlerin belirli korelasyonla çıkıp çıkmadığına bakıyoruz
# eksikliklerin bağımlı olma durumu ve birlikte çıkma durumu
# iki değişkenin birlikte eksiliyor olması
# +1 e yakın olması pozitif yönlü kuvvetli ilişki -> eksiklikler birlikte ortaya çıkmıştır
# -1 e yakın olması negatif yönlü kuvvetli ilişki -> birisinde yokken diğerinde vardır ters ilişki

# bu veri seti anlamlı gözükmüyor -1 e yakın

msno.heatmap(df)
plt.show(block=True)


#########################################

# Eksik Değerlerin Bağımlı Değişken İle İlişkisinin İncelenmesi

#########################################

# amacım : bir veri seti var ve bu veri setinde eksiklikler var,acaba bu eksikliklerin bağımlı değişkenlerle ilişksi var mı
# yoksa özel bir yapı mı ?


missing_values_table(df,True)
na_cols = missing_values_table(df,True)

def missing_vs_target(dataframe,target,na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(),1,0)
    na_flags = temp_df.loc[:,temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN":temp_df.groupby(col)[target].mean(),
                            "Count":temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df,"Survived",na_cols)

# dipnot : çalışanların kabin numarası yok sadece yolcuların var

# target ilgili veri değişkenindeki bağımlı değişken -> survived
# eksiklik varsa eksiklik gördüğün yere 1 diğerine 0
# temp i oluşturma amacım orjinal dataframemi bozmamak için copy oluşturdum


###############################

# RECAP

###############################

df = load()

na_cols = missing_values_table(df,True)
# sayısal değişkenleri direk median ile oldurma
df.appy(lambda x:x.fillna(x.median()) if x.dtype != "0" else x , axis = 0).isnull().sum()

# kategorik değişkenleri mode ile doldurma
df.appyl(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "0" and len(x.unique()) <= 10)else x, axis=0).isnull().sum()

# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

# tahmine dayalı atama ile doldurma

missing_vs_target(df,"Survived",na_cols)

