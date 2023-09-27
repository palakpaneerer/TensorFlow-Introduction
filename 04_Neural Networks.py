# TensorFlow and tf.keras　画像データ
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# データセットのロード
fashion_mnist = keras.datasets.fashion_mnist  
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# データの確認
print(train_images.shape)
#->60000個の画像データ。各データ28×28の大きさ
 
#0個目の画像の23個目×23個目のピクセルの値を確認
print(train_images[0,23,23])
#->0（真っ黒）~255（真っ白）の大きさで表せる。今回は194なのでグレー

#train_labelsは今回0-9の値で格納されてれいる。
#それぞれ、どのようなアイテムか事前に以下のclass_namesの様に決められている。
#最初の10個アイテムの確認
print(train_labels[:10])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#画像データ確認
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

#前処理
#すべてのグレースケール ピクセル値(0~255)を0~1の間になるように単純にスケーリングします。
#これは値が小さいほどモデルによる値の処理が容易になるため。
train_images = train_images / 255.0
test_images = test_images / 255.0

#モデルの設定
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # インプットレイヤー (1)
    keras.layers.Dense(128, activation='relu'),  # ヒドゥンレイヤー　(2)
    keras.layers.Dense(10, activation='softmax') # アウトプットレイヤー (3)
])

#モデルをコンパイル
#損失関数、オプティマイザー、およびメトリクスを定義
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#モデルにフィットさせる
model.fit(train_images, train_labels, epochs=10) 

#テストデータセットを使って、モデルの評価
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
print('Test accuracy:', test_acc)


#テストデータの予測値をpredictionsに格納
predictions = model.predict(test_images)
#画像個目の値を確認
print(predictions[0])
#->9個目アイテムのカテゴリーの可能性をそれぞれ数値で教えてくれる。
#一番数の大きいもに該当するアイテム項目が、写真のものだと予想されている。

#argmaxにて一番大きい数のインデックスを取得
print(np.argmax(predictions[0]))
#->9　つまり'Ankle boot'

#正解データと照らし合わせる。
print(test_labels[0])
#->9　つまり'Ankle boot'で正解。


#検証
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.clf()
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.colorbar()
  plt.grid(False)
  plt.show()
  print("Excpected: " + label)
  print("Guess: " + guess)

def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)







