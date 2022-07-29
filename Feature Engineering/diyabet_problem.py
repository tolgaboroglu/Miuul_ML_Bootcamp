import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt



import missingno as msno
from datetime import date

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.3f'%x)
pd.set_option('display.width',500)

def load():
    data = pd.read_csv('datasets/diabetes.csv')
    return  data


####################################### GÖREV 1 KEŞİFÇİ VERİ ANALİZİ ############################################



# Adım 1 : genel resmin incelenmesi

df = load()
df.shape
df.head()
df.info
df.tail()
df.isnull().sum

# genel olarak yazılması

def check_df(df, head=5):
    print("####################### shape ##########################")

    print(df.shape)

    print("####################### types ##########################")

    print(df.dtypes)

    print("####################### head ##########################")

    print(df.head)

    print("####################### tail ##########################")

    print(df.tail)

    print("####################### NA ##########################")

    print(df.isnull().sum)

    print("##################### Quantiles #####################")

    print(df.quantile([0,0.05,0.50,0.95,0.99,1]).T)
check_df(df)

df

# Adım 2 : Numerik ve kategorik değişkenlerin yakalanması

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

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "0"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                dataframe[col].dtypes != "0"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                dataframe[col].dtypes == "0"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations:{dataframe.shape[0]}")
    print(f"Variables:{dataframe.shape[1]}")
    print(f'cat_cols:"{len(cat_cols)}')
    print(f'num_cols:"{len(num_cols)}')
    print(f'cat_but_car:"{len(cat_but_car)}')
    print(f'num_but_cat:"{len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3 : Numerik ve kategorik değişkenlerin analizinin yapılması

num_cols

cat_cols

# programatik şekilde yazalım

# numeric

def numSummary(dataframe, numericalCol, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
    print(dataframe[numericalCol].describe(quantiles).T)

    if plot:
        dataframe[numericalCol].hist()
        plt.xlabel(numericalCol)
        plt.title(numericalCol)
        plt.show(block=True)


for col in num_cols:
    print(f"{col}:")
    numSummary(df, col, True)

# categorical

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    if df[col].dtypes == "bool":
        print(col)
    else:
        cat_summary(df, col, True)



# Adım 4 : Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n\n\n")

for col in num_cols:
        target_summary_with_num(df,"Outcome",col)



# Adım 5 : Aykırı gözlem analizinin yapılması

def outliers_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit , up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outliers_thresholds(dataframe, col_name)
    # aslında yukarıada yaptığımız any yani bool ile herhangi bir boş aykırı değer var mı sorusuna denk gelir
    if dataframe[(dataframe[col_name]>up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

    check_outlier(df, "Outcome")

for col in num_cols:
    print(col, "=>", check_outlier(df, col))


def grab_outliers(dataframe, col_name, index = False):
    low,up = outliers_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name]>low) | (dataframe[col_name] < up)].shape[0] > 10:
        print(dataframe[(dataframe[col_name]<low)| (dataframe[col_name] > up)].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name]> up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name]< low) | (dataframe[col_name] > up))].index
        return outlier_index

    for col in num_cols:
        print(col, grab_outliers(df, col, True))


# Adım 6 : Eksik gözlem analizinin yapılması

def missing_values_table(dataframe,na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss,np.round(ratio,2)], axis=1, keys=['n_miss','ratio'])
    print(missing_df,end="\n")

    if na_name:
        return na_columns

missing_values_table(df)



# Adım 7 : Korelasyon analizinin yapılması

# kolerasyon hesaplamak için "corr" fonksiyonu kullanılır ;

corr = df[num_cols].corr()
corr

# ısı haritası oluşturalım ;

sns.set(rc={'figure.figsize':(12,12)})
sns.heatmap(corr,cmap="RdBu")
plt.show(block=True)


##########################

# BASE MODEL KURULUMU

##########################

y = df["Outcome"]
X = df.drop("Outcome",axis = 1)
X_train,X_test ,y_train,y_test = train_test_split(X,y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train,y_train)

# hiçbir işlem yapmadan base model kuracağız



######################################### FEATURE ENGINEERING ##################################################


# Adım 1 : Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.


for col in zero_columns :
    df[col] = np.where(df[col] == 0 , np.nan,df[col])




# Adım 2: Yeni değişkenler oluşturunuz.




# Adım 3: Encoding işlemlerini gerçekleştiriniz.



# Adım 4: Numerik değişkenler için standartlaştırma yapınız.





# Adım 5: Model oluşturunuz.