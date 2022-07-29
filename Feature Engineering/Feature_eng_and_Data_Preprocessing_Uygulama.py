import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt



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
df.shape

df.head()
############################################## AYKIRI DEĞERLER I DEF ETME ############################################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "0"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique()<cat_th and
            dataframe[col].dtypes !="0"]
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
    return cat_cols,num_cols,cat_but_car
cat_cols,num_cols, cat_but_car = grab_col_names(df)

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outliers_thresholds(dataframe, col_name)
    # aslında yukarıada yaptığımız any yani bool ile herhangi bir boş aykırı değer var mı sorusuna denk gelir
    if dataframe[(dataframe[col_name]>up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def outliers_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit , up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outliers_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable]<low_limit),variable] = low_limit
    dataframe.loc[(dataframe[variable]>up_limit),variable] = up_limit

####################################################################################################################################

################################################### EKSİK DEĞERLERİ DEF ETME #######################################################

def missing_values_table(dataframe,na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss,np.round(ratio,2)], axis=1, keys=['n_miss','ratio'])
    print(missing_df,end="\n")

    if na_name:
        return na_columns



############################################################################################################################

# değişken isimleri büyük ve küçük harften oluşuyor bunların hepsini tek bir formata getirelim

df.columns = [col.upper() for col in df.columns]
df.head()



###################################################### LABEL ENCODER DEF I #################################################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform((dataframe[binary_col]))
    return  dataframe

#############################################################################################################################

###################################################### ONE HOT ENCODER DEF I ################################################

def one_hot_encoder(dataframe, categorical_cols, drop_first = False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols,drop_first=drop_first)
    return dataframe

#############################################################################################################################

###################################################### RARE ENCODING DEF I ##################################################

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col,":",len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN":dataframe.groupby(col)[target].mean()}),end="\n\n\n")



#############################################################################################################################

###################################################### RARE ENCODER DEF I #########################################################

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

########################################

# 1.Feature Engineering (Değişken Mühendisliği)

########################################

df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int64')

df["NEW_NAME_COUNT"] = df["NAME"].str.len()

df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))

df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df["NEW_TITLE"] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1

df.loc[((df["SIBSP"] + df["PARCH"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SIBSP"] + df["PARCH"]) == 0), "NEW_IS_ALONE"] = "YES"

df.loc[(df["AGE"] < 18), "NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] >= 56), "NEW_AGE_CAT"] = "senior"

df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 21) & (df["AGE"] <= 50), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniormale"
df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 21) & (df["AGE"] <= 50), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniorfemale"


########################################

# 2.Outliers ( Aykırı Değerler)

for col in num_cols :
    print(col, check_outlier(df,col))

for col in num_cols:
    replace_with_thresholds(df,col)


# tekrar aykırı değerlere bakalım

for col in num_cols:
    print(col, check_outlier(df,col))



###############################################

# 3.Missing Values ( Eksik Değerler )

################################################

missing_values_table(df)
# silelim

df.drop("CABIN",inplace = True, axis=1)
missing_values_table(df)


# tekrar silelim remove cols ile

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)
missing_values_table(df)

# yaş değişkenin eksik değerlerini median ile dolduralım / new title a göre

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
missing_values_table(df)


# peki diğer yaşa bağlı değişkenler ne olacak ? / tekrar baştan oluşturacağız

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df["AGE"] < 18), "NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] >= 56), "NEW_AGE_CAT"] = "senior"

df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 21) & (df["AGE"] <= 50), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniormale"
df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 21) & (df["AGE"] <= 50), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

missing_values_table(df)

# son kalan embarkedı da silerek kurtulduk

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
missing_values_table(df)


#########################################

# 4.Label Encoding

########################################

# eşsiz 2 sınıfa sahip olan 2 değişkeni seç

binary_cols = [col for col in df.columns if df[col].dtype not in[int, float]
                and df[col].nunique() == 2]

binary_cols

for col in binary_cols:
    df = label_encoder(df,col)

binary_cols

############################################

# 5. Rare Encoding

############################################

rare_analyser(df,"SURVIVED",cat_cols)

df = rare_encoder(df, 0.01)

df.head()

df["NEW_TITLE"].value_counts()


#############################################

# 6.One-Hot Encoding

#############################################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)
df

df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

num_cols

# amacım yeni oluşturduğum değişkenlerin frekansları birbirine yakın olsun istiyorum
# ne kadar anlamlı bakıyoruz

rare_analyser(df, "SURVIVED",cat_cols)

# kullanışsız sütunları filtreliyoruz değeri düşük çok bir anlam ifade etmiyor
# değeri 0.01 den düşük olanları getir diyoruz

useless_cols = [col for col in df.columns if df[col].nunique()  == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

useless_cols

#bunları silebiliriz

df.drop(useless_cols, axis=1,inplace=True)

df.head()

#########################################

# 7. Standard Scaler (Standartlaştırma)

########################################
#not : bu problemde gerekli değil ama eğer ihtiyacımız olursa kullanırız

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()


##########################################

# 8. Model

##########################################

# bağımlı ve bağımsız değişkenleri seçmem gerekiyor
# biriyle test diğeriyle train edilsin

y = df["SURVIVED"]
X = df.drop(["PASSENGERID","SURVIVED"],axis=1)

X_train ,X_test, y_train , y_test = train_test_split(X,y,test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


########################################

# Hiçbir işlem yapılmadan elde edilecek skor

#########################################

dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex","Embarked"],drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId","Survived","Name","Ticket","Cabin"],axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
rf_model = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


# yeni ürettiğimiz değişkenler ne alemde ?

def plot_importance(model, features, nun = len(X), save = False):
    features_imp = pd.DataFrame({'Value':model})
