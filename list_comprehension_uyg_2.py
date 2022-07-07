### List  & Dictionary Comprehension Uygulama 2

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

[col for col in df.columns if "INS"in col]

["FLAG" + col for col in df.columns if "INS" in col]

["FLAG" + col if "INS" in col else "NO_FLAG_"+  col for col in df.columns]

df.columns = ["FLAG" + col if "INS" in col else "NO_FLAG_"+  col for col in df.columns]

