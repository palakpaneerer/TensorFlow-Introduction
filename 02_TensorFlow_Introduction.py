# Module 2: Introduction to TensorFlow
# URL: https://www.youtube.com/watch?v=tPYj3fFJGjk&list=LL&index=1

import tensorflow as tf
print(tf.version)

# 変数の定義と型指定
string = tf.Variable("this is a string", tf.string) 
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

# テンソルの定義と型指定
rank1_tensor = tf.Variable(["Test"], tf.string) 
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)

# print(f'rank1_tensorのランクは{tf.rank(rank1_tensor)}です。')
# print(f'rank1_tensorの詳細は{rank1_tensor}です。')
# print(f'rank1_tensorの形状は{rank1_tensor.shape}です。')

# print(f'rank2_tensorのランクは{tf.rank(rank2_tensor)}です。')
# print(f'rank2_tensorの詳細は{rank2_tensor}です。')
# print(f'rank2_tensorの形状は{rank2_tensor.shape}です。')

# テンソルの形状の変更
tensor1 = tf.ones([1,2,3]) 
tensor2 = tf.reshape(tensor1, [2,3,1])  
tensor3 = tf.reshape(tensor2, [3, -1]) #-1は他項目に合わせて調整するという意味

# print(tensor1)
# print(tensor2)
# print(tensor3)

# 2次元配列の作成
matrix = [[1,2,3,4,5],
          [6,7,8,9,10],
          [11,12,13,14,15],
          [16,17,18,19,20]]

tensor = tf.Variable(matrix, dtype=tf.int32) 
print(tf.rank(tensor))
print(tensor.shape)


# テンソル内の値を選択
three = tensor[0,2] #縦:(0+1)個目、横：(2+1)個目 
print(three)  # -> 3

row1 = tensor[0]  #縦:(0+1)個目、横:全部
print(row1)

column1 = tensor[:, 0] #縦:全部、横:(0+1)個目
print(column1)

row_2_and_4 = tensor[1::2] #縦:2,4列目、横:全部
print(row_2_and_4)

column_1_in_row_2_and_3 = tensor[1:3, 0] #縦:2,3列目、横:1個目
print(column_1_in_row_2_and_3)




# テンソルの作成、変形、表示
t = tf.zeros([5,5,5,5])
t = tf.reshape(t,[125,-1])
print(t)

