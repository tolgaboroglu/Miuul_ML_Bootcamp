###################################

# Encoding

###################################

# Ne Demek ?

# Değişkenlerin temsil şekillerinin değiştirilmesi
# Label Encoding:  -> yeniden kodlamak temsil ettirmek
# sex : male-female-male-male-female
# is_female :  0-1 male : 0 female : 1

# farklılığa göre label encoding oluyor sınıflar arası farklılıklar olmalı
# yaş-cinsiyet-ünvan vs


###############################################

# 3.Encoding (Label Encoding, One-Hot Encoding, Rare Encoding )

###############################################

###############################################
# Label Encoding & Binary Encoding
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
df["Sex"].head()

# cinsiyeti 0-1 olarak kodlayalım

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]

# ilk gördüğü değere 0 verir(alfabetik sıraya göre 0 -1 )

# hangi sınıf olduğunu unuttuk hangisi 0 hangisi 1

le.inverse_transform([0,1])

# bu tercih edilir/fonksiyon yazılır

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform((dataframe[binary_col]))
    return  dataframe

df = load()

# yüzlerce veri varsa binary_collar seçilir

# 2 sınıflı kategorik değişkenleri label_encoderdan geçireceğiz

binary_cols = [col for col in df.columns if df[col].dtype not in [int,float]
            and df[col].nunique() == 2]

binary_cols


# diyelim ki 10 tane vardı

for col in binary_cols:
    label_encoder(df,col)

binary_cols

df.head()

# application_train_csv için de yapalım

def load_application_train():
    data = pd.read_csv('datasets/application_train.csv')
    return data

df = load_application_train()
df.shape

df.head()

binary_cols = [col for col in df.columns if df[col].dtype not in [int,float]
            and df[col].nunique() == 2]
binary_cols

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df,col)

df[binary_cols].head()

# na ların farkında ol yani 2 ler
df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique())

##########################################

# 4.One Hot Encoding

##########################################

# sınıflar arası fark yokken varmış gibi olacaktır
# gs-ts-fb-bjk aynı -> ben bunları öyle temsil etmeliyim ki durduk yere pek bir fark koymiyim
# sınıfları değişkenlere dönüştürme
# GS = 1 ise diğerleri 0 gibi -> kukla değişkendir / ölçme problemi çıkabilir birbiri üzerinde oluşturabilir
# ilk sınıf drop edilirse problem yaşanmaz


df = load()
df.head()
df["Embarked"].value_counts()

# get_dummies -> bana bir dataframe söyle ben sadece onları dönüştürecem/diğerleri olduğu gibi kalacak
pd.get_dummies(df,columns=["Embarked"]).head()

# drop_first ü kullanacağız dummi ile birbiri üstüne geçmemesi için

pd.get_dummies(df,columns=["Embarked"], drop_first=True).head()

# eğer ilgili değişkendeki eksik değerlerde gelsin istersek

pd.get_dummies(df,columns=["Embarked"], dummy_na=True).head()

#hepsini tek bir adımda yapmak istersek

pd.get_dummies(df,columns=["Sex","Embarked"], drop_first=True).head()

# fonksiyonlaştıralım

def one_hot_encoder(dataframe, categorical_cols, drop_first = False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols,drop_first=drop_first)
    return dataframe

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

cat_cols,num_cols, cat_but_car = grab_col_names(df)

# yorum : hepsini geçir "survived" dışında kalsın

# one_hot_encoder dan geçecek olan sütunları da seçerek takip edebilirim****
# 2 den büyükse 10 dan küçükse
# bağımlıysa dışarda bırak

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2 ]

ohe_cols

one_hot_encoder(df,ohe_cols).head()

# kalıcı değil
df.head()

#kalıcı olması için = df = one_hot_encoder(df,ohe_cols).head()


##########################################

# Rare Encoding

##########################################

# Ne demek ?

# az gözlemlenen

# 2-3-5 gibi az gözlemler var bunları napacağız
# büyük çoğunluğu temsil etmek amacımız
# nadir değerleri almayız bize doğru bilgiyi vermez çünkü gözlem sayısı az
# ölçümlerin de kalitesi olsun / gereksiz birçok değişken oluşturmak istemiyoruz
# gereksiz değişkenlerden uzaklaşmamız gerek bu yüzden kullanırız
# tüm işlemlerde kullanılmaz

# Nasıl yapacağız

# bir eşik değeri belirleyeceğiz bunların altındakileri işaretleyeceğiz -> -> RARE diyeceğiz(altındakilere)
# diğer kalanları bir araya getireceğiz



# Uygulaması

# 1.Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi
# 2.Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi
# 3.Rare encoder yazacağız


##########################################

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi

##########################################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

# az olan birden fazla sınıfı bir araya getirmekle ilgileniyoruz

# kaç kategorik değişken var buna erişmek istiyorum

df.head()

# nasıl seçeceğiz hepsini kategoriklerin
# grab_col_names ile

cat_cols, num_cols,cat_but_car = grab_col_names(df)


def cat_summary(dataframe,col_name,plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##################################################")

    if plot:
        sns.countplot(x= dataframe[col_name],data=dataframe)
        plt.show(block= True)

for col in cat_cols:
    cat_summary(df, col)


#########################################

# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi

#########################################

df["NAME_INCOME_TYPE"].value_counts()

# 0 yakın olması krediyi ödeyebilmeyi ifade etmektedir

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

# yukardaki iki işlemi bir araya getirerek kolaylık sağlarız/ elimizdeki tüm kategorik değişkenler için uyguladık

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col,":",len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN":dataframe.groupby(col)[target].mean()}),end="\n\n\n")

rare_analyser(df,"TARGET",cat_cols)

#####################################

# 3. Rare encoder'ın yazılması

####################################

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == '0'
                    and (temp_df[col].value_counts() / len(temp_df) <rare_perc).any(axis=None)]
# toplam gözlem sayısına bölünüyor
    for var in rare_columns:
        tmp = temp_df[var].value_counts()  / len(temp_df)
        # hepsini bir araya getirip ismini "rare" yapmak istiyorum.
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels),'Rare',temp_df[var])

    return  temp_df

# new_df diyerek tüm kategorik değişkenleri encoder dan geçirdim
new_df = rare_encoder(df,0.01)

rare_analyser(new_df, "TARGET",cat_cols)


######################################################

# Feature Scaling (özellik ölçeklendirme)

######################################################


# Nedir ?

# amacımız değişkenler arasındaki ölçüm farklılığını gidermektedir.
# kullanılacak olan modelin eşit şartlarda yarışmasıdır
# scale edilmiş featureslar ölçeklenmiş boyutların min noktaya en hızlı şekilde ulaşır/trailer sürelerini,eğitim sürelerini kısaltmaktır
# yapacağımız işlem standartlaştırmak
# ağaca dayalı yöntem değerden etkilenmez


#############################

# StandardScaler : Klasik standartlaştırma.Ortalamayı çıkar,standart sapmaya böl. z=(x -u ) / s

#############################

df = load()
ss = StandardScaler()
# kıyaslama yapmak için
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])

df.head()


#######################

# RobustScaler : Medyanı çıkar iqr'a böl.

#######################
# daha çok tercih edilir,aykırı değerlere karşı daha dirençli

rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

# robust scaler , standart scaler a göre ortalaması farklı çünkü robust scaler daha az etkileniyor


#####################

# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü

#####################

# özellikle istediğimiz bir aralaık varsa bu yöntem kullanılır

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

# kıyaslayalım

age_cols = [col for col in df.columns if "Age" in col]

age_cols

#görselleştirme

def num_summary(dataframe,numerical_col, plot=False):
    quantiles = [0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

# yapmak istediğim ortaya çıkan gözlemlerde herhangi bir değişiklik var mı onu göstermek istiyorum

for col in age_cols:
    num_summary(df,col,plot=True)

plt.show(block=True)

# sonuç : yapılarını koruyacak şekilde ifade tarzlarını değiştirdik
# dağılım bozulmadı



#########################

# Numeric to Categorical : Sayısal Değişkenleri Kategorik Değişkenlere Çevirme
# Binning

#########################
#yaş/qcut
# kaç parça olacak = 5
# label olsa girecektik
# k den b ye sıralar ve çeyrek değerlere böler

df["Age_qcut"] = pd.qcut(df['Age'],5)

df.head()

