from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score

import numpy as np

from keras.utils import to_categorical

import pickle as pk

import pandas as pd

# Import data
pic_bert = open("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/Math/mydata/level/level_pooled_bert.pickle", "rb")

x = pk.load(pic_bert)

# get tags
y = pd.read_excel("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/Math/mydata/level/level_preprocess.xlsx")

xtrain, xvalid, ytrain, yvalid = train_test_split(x, y["tags"], random_state=42, test_size=0.2)

print(ytrain[:5])
ytrain = to_categorical(np.array(ytrain))

print(ytrain[:5])



# pic_out_xtrain = open("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/Math/mydata/level/XGBoost_Data/xtrain.pickle", "wb")
# pic_out_xvalid = open("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/Math/mydata/level/XGBoost_Data/xvalid.pickle", "wb")
# pic_out_ytrain = open("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/Math/mydata/level/XGBoost_Data/ytrain.pickle", "wb")
# pic_out_yvalid = open("C:/Users/Neil Marcus/Desktop/SRP Project/Math/Math/Math/mydata/level/XGBoost_Data/yvalid.pickle", "wb")
#
# pk.dump(xtrain, pic_out_xtrain)
# pk.dump(xvalid, pic_out_xvalid)
# pk.dump(ytrain, pic_out_ytrain)
# pk.dump(yvalid, pic_out_yvalid)
#
# pic_out_ytrain.close()
# pic_out_xvalid.close()
# pic_out_ytrain.close()
# pic_out_yvalid.close()
#
# xgboost_clf = XGBClassifier(silent=0, learning_rate=0.1)
# xgboost_clf.fit(xtrain, ytrain)
#
# predict = xgboost_clf.predict(xvalid)



