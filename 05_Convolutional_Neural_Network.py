# Convolutional neural network 画像データ

#アーキテクチャの概要　大きく3part
#part1
#畳み込み層：画像データ*フィルター（各要素を掛けた後合計）→特徴マップ
#プーリング層：特徴マップにmaxpooling等でさらに強い特徴をピックアップ→ダウンサイズマップ

#part2
#Dence（濃縮）層：ニューラルネットワークを使用。

#part3
#全結合層：1次元データのクラスタリング予測値を返す。ex.犬、猫等


#今回使うデータラベル
# Airplane
# Automobile
# Bird
# Cat
# Deer
# Dog
# Frog
# Horse
# Ship
# Truck

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

#データセットをロード
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

#前処理
#全ての値を0~1にする。
train_images, test_images = train_images / 255.0, test_images / 255.0

#画像データのクラスを明示
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

#画像データを実際に確認 
IMG_INDEX = 1  
plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()


#CNNアーキテクチャー、畳み込み層&Pooling層
model = models.Sequential()
#畳み込み層(Conv2D)　3×3の小さな32個のフィルターを用いて異なる特徴を検出していく。
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
#Pooling層ではMax関数が使用されている。2×2の正方形の中で、最大値を抽出していく。
model.add(layers.MaxPooling2D((2, 2)))
#さらなる畳み込み層
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#さらなるPooling層
model.add(layers.MaxPooling2D((2, 2)))
#さらなる畳み込み層
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#CNNアーキテクチャーのサマリーを確認
print(model.summary())


#CNNアーキテクチャー、Dence（濃縮）層、全結合層
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
#全結合層:10種類のカテゴリーに対してそれぞれの可能性%を返す。
model.add(layers.Dense(10))


print(model.summary())

#トレーニング
#losses.SparseCategoricalCrossentropyは多クラス分類タスクにおける損失関数
#from_logits=Trueは夫々のカテゴリーに確率値を返してくれるように設定している。
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4, 
                    validation_data=(test_images, test_labels))


#モデルの評価
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#verbose=0: 進捗状況を出力しない。
#verbose=1: 進行状況バーを表示する。
#verbose=2: 評価の進行状況バーは表示せず、評価の結果だけを表示する。

print(test_acc)







