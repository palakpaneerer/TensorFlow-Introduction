# Linear Regression

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

print(tf.version)

# データセットのロード　タイタニック
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived') #.popは抽出。抽出元のデータフレームからは除外される。
y_eval = dfeval.pop('survived')

# EDA
print(dftrain.head())
print(dftrain.describe())
print(dftrain.shape)
print(y_train.head())

dftrain.age.hist(bins=20)
dftrain.sex.value_counts().plot(kind='barh')
dftrain['class'].value_counts().plot(kind='barh')
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')


# 型の指定
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# 特徴量を機械学習の学習用データに使用できるように修正
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

# データセットの作成について設定する関数
# バッチサイズはメモリ量を考慮して調整する必要がある。
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=16):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) 
    if shuffle:
      ds = ds.shuffle(1000)  
    ds = ds.batch(batch_size).repeat(num_epochs)  
    return ds 
  return input_function  

# 関数を使って、学習データセットと正解データセットの設定
train_input_fn = make_input_fn(dftrain, y_train) 
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# 予測の実行
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

#  予測結果の精度確認
linear_est.train(train_input_fn)  
result = linear_est.evaluate(eval_input_fn) 

clear_output() 
print(result['accuracy'])



pred_dicts = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[0])
print(y_eval.loc[0])


pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

plt.clf() # 前のグラフをクリアしておく。
probs.plot(kind='hist', bins=20, title='predicted probabilities')



