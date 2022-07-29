#################################

# Feature Extraction (Özellik Çıkarımı)

#################################

# Ne demek ?

# ham veriden değişken üretmek
# 2 tiptir :
# - yapısal verilerden değişken türetmek
# - yapısal olmayan verilerden değişken türetmek

# Yapısal değişken : elimizdeki mevcut değişkenden türetmek
# Yapısal olmayan : görüntü - sesten veri türetmek

# yapısal olmayan anlamsız gibi görünen verilerden de bir anlam ifadesi çıkarılabilir
# bağımlı değişkenli durumlarını göz ardı etmeyelim

# örn : doğal dil işleme teknikleriyle filmin tanımını yani string bir ifadeyi tavsiye sistemine yönelik film öneri sistemi
# geliştirmek istiyoruz

# filmin açıklamalarında yer alan bütün eşssiz kelimeleri sıralamalara alırız  / sayısal değerler atarız
# bu filmde bu kelimeden kaç tane var gibi benzer filmlerin tanıtımlarına bakarak kıyas yaparız  / kaç defa geçmiş
# math formatinda yazacağız

# kedi örn : kedi resmi üzerinde çalışmak istiyorum
# öyle bir şey yapmalıyım ki lineer cebir şekle getirmeliyim
# pixellerin yoğunluklara göre bir sayıyla ifade etmeliyim ki inceleyeyim



##########################

# Binary Features : Flag, Bool, True-False

##########################
# var olan değişkenler üzerinden yeni değişkenler türetmek
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

# NA = 0 -> boş değerler
# Diğerleri/Dolu = 1 yazalım bir gözlemleyelim

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

df.head()

# bağımlı değişkene göre ortalamayı alalım
df.groupby("NEW_CABIN_BOOL").agg({"Survived":"mean"})

# sonuç: kabini dolu olanların hayatta kalma oranları : "1" ve daha yüksektir

# bu gerçekten bir ilişki taşıyor mu ?
# oran testi yaparak sağlamasını yapalım

from statsmodels.stats.proportion import proportions_ztest

test_stat,pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                            df.loc[df["NEW_CABIN_BOOL"]  == 0, "Survived"].sum()],

                                    nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                        df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = % 4f, p-value = %4f' % (test_stat,pvalue))

# p - value değeri 0.05 den küçük olduğundan değeri ikisi arasında farklılıklar vardır .


# gemideki akrabalıkları ifade eder yani bu kişi gemide yalnız değilmiş
# eğer yalnızsa -> yes , değilse -> no
# hayatta kalma çabasını etkileyebilir yalnız olmaması

df.loc[(df['SibSp'] + df['Parch'] + df['Parch'] > 0) , "NEW_IS_ALONE"] = "NO"
df.loc[(df['SibSp'] + df['Parch'] + df['Parch'] == 0), "NEW_IS_ALONE"] = "YES"

#sonuç : yalnız olanların hayatta kalma oranları daha düşük
df.groupby("NEW_IS_ALONE").agg({"Survived":"mean"})

#hipotez testi yapalım

test_stat,pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                            df.loc[df["NEW_IS_ALONE"]  == "NO", "Survived"].sum()],

                                    nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                        df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = % 4f, p-value = %4f' % (test_stat,pvalue))


# sonuç : hayatta kalma durumunu kesin olarak göstermez ama göz ardı edilemeyeceğini gösterir
#         daha net etkisi modellemede farkedilir



#####################################################

# Text'ler Üzerinden Özellik Türetmek

#####################################################

df.head()

# mesala bir veri setindeki metinlerden değişkenler türetelim

###################

# Letter Count

###################

# namedeki ifadeleri say

df["NEW_NAME_COUNT"] = df["Name"].str.len()

# bir değişkende kaç tane harf var saydırır

df.head()

#####################

# Word Count

#####################
# ismi yakalayım stringe çevirsin
# sonra split et ve kaç tane varsa say

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

df.head()

####################

# Özel Yapıları Yakalama

####################

#dipnot: mümkün olduğunca az değişkenle yüksek başarı elde etmek istiyoruz
# dr ifadesini seç
df["NEW_NAME_DR"] = df["Name"].apply(lambda  x: len([x for x in x.split() if x.startswith("Dr")]))

df

# df.head veya df dediğimizde gözlemleyemeyiz o yüzden dr ye göre groupby yapacağız
# survived a göre mean al
# sonuç : dr olanların hayatta kalma oranları daha yüksektir

df.groupby("NEW_NAME_DR").agg({"Survived" : "mean"})


# kaç dr var

df.groupby("NEW_NAME_DR").agg({"Survived":["mean","count"]})

########################################

# Regex Features ile Değişken Türetmek

########################################


df.head()

# pattern / örüntüyü anlamamız lazım
# metinler arasındaki
# MR MS gibi hepsinde aynı gibi
# büyük ya da küçük harflerden oluşacak diye ayarlayacam


df['NEW_TITLE']  = df.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

df.head()


# groupby yapacağız

df[["NEW_TITLE", "Survived","Age",]].groupby(["NEW_TITLE"]).agg({"Survived":"mean", "Age": ["count","mean"]})

# sonuç : yaş ortalamasını sınıf bazında göreceğiz
#         veri değişken üzerinden yeni değişken türettim , bu kategorik değişken üzerinden yeni değişken türetmek daha mantıklı


#######################################

# Date Features

########################################
#kurstaki puanlamalar

dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()

# amacımız "Timestamp" değişkeninden bir veri türetmek
# bir problemimiz var değiştirmek istediğimiz değişken object biçiminde
# önce bu değişkenin türünü değiştirelim "to_datetime" kullanarak

dff["Timestamp"] = pd.to_datetime(dff["Timestamp"], format= "%Y-%m-%d")
dff.head()

dff.info()

# year

dff['year'] = dff['Timestamp'].dt.year

# month

dff['month'] = dff['Timestamp'].dt.month

# day

dff['day'] = dff['Timestamp'].dt.day


# year diff
# günümüzden çıkar

dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year


# iki tarih arasındaki farkı ay cinsinden ele alalım
# moth diff (iki tarih arasındaki ay farkı) : yıl farkı + ay farkı

dff['moth_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month

dff.head()

# day name

dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head()


############################################

# Features Interactions (Özellik Etkileşimleri)

############################################

# Ne demek ?

# değişkenlerin birbirleriyle ilişkide olması demek
# bir değişkenin çarpılması,toplanması,karesinin alınması,küpü........ vs

# örn
# yaş değişkeni ile pclassları çarpalım
# yaş değişkenin pclass ile arasındaki ilişkiye bakalım/refah durumu

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df.head()


# mesala akraba sayıları ve ilişkileri +1 kendisi dersek

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
df.head()

# belirli kategorik ve sayısal değişkenlerin etkileşimleri

# yaşı 21 den küçük olan erkekler

df.loc[(df["Sex"] == 'male') & (df['Age'] < 21), 'NEW_SEX_CAT'] = 'youngmale'


#yaşı 21 den büyük olan erkekler

df.loc[(df["Sex"] == 'male') & (df['Age'] > 21), 'NEW_SEX_CAT'] = 'maturemale'

# yaşı 50 den büyük olan erkekler

df.loc[(df["Sex"] == 'male') & (df['Age'] < 21), 'NEW_SEX_CAT'] = 'seniormale'


#aynı işlemleri kadınlar için de yapıyoruz

# yaşı 21 den küçük olan kadınlar

df.loc[(df["Sex"] == 'female') & (df['Age'] < 21), 'NEW_SEX_CAT'] = 'youngfemale'

#yaşı 21 den büyük olan kadınlar

df.loc[(df["Sex"] == 'female') & (df['Age'] > 21), 'NEW_SEX_CAT'] = 'maturefemale'

# yaşı 50 den büyük olan kadınlar

df.loc[(df["Sex"] == 'female') & (df['Age'] < 21), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()

# analiz/ çıkarım aşaması
# yaş ve cinsiyet bazında hayatta kalanların yaş aralık ortalaması

df.groupby("NEW_SEX_CAT")["Survived"].mean()