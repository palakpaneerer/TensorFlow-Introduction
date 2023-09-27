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


# 特徴量を機械学習の学習用データに使用できるように修正する関数
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


# トレーニングデータセットの作成
# データを数値であると教えてあげ、my_feature_columnsに格納していく。
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)


# Deep Neural Network (DNN) ：2 hidden layers（30 and 10 hidden nodes)を作成.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # 隠れ層設定
    hidden_units=[30, 10],
    # アウトプットは3種類のどれか。
    n_classes=3)


# 訓練
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

# エポック（Epoch）: 1つのエポックは、訓練データセット全体を一度処理することを指します。
# バッチ（Batch）: データセットを小さな塊に分割したもの。
# ステップ（Step）: バッチ単位でのデータ処理の回数を指します。
# 全体のデータが32で、バッチサイズが32の場合、64ステップすると2エポック


# 評価
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))
# training=Falseは評価データのためシャッフルしない旨を伝えている
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))



# 参考
# 入力したデータの予測とその予測精度に対する自信の言及
def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid: 
    val = input(feature + ": ")
    if not val.isdigit(): valid = False

  predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))






