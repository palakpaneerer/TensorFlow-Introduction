# Classification

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


#　データセットのロード
train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# データフレームの確認
train.head()

# yデータの抽出
train_y = train.pop('Species')
test_y = test.pop('Species')


# 特徴量を機械学習の学習用データに使用できるように修正
def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

#shuffle=True: データセット内のデータをシャッフルするかどうかを制御します。
#具体的には、データがミニバッチに分割される前にデータの順序をランダムに変更します。
#シャッフルはトレーニングデータに適用され、トレーニング中にモデルがデータの偏りを学習しないようにするために重要です。

#training=True: これはデータセットをトレーニングモードか評価モードかに設定します。
#トレーニングモードでは、データセットはシャッフルされ、エポックがリピートされます。
#評価モードでは、データセットはシャッフルされず、エポックは一度だけ実行されます。

#したがって、shuffle=True はデータのシャッフルに関連し、トレーニングモードでのみ適用されますが、
#training=True はデータセット全体のトレーニングまたは評価モードを制御します。


# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)



