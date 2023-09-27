# Clustering(隠れマルコフモデル)

import tensorflow_probability as tfp #regression と classificationと違うモジュール
import tensorflow as tf

"""
Cold days are encoded by a 0 and hot days are encoded by a 1.

The first day in our sequence has an 80% chance of being cold.

A cold day has a 30% chance of being followed by a hot day.

A hot day has a 20% chance of being followed by a cold day.

On each day the temperature is normally distributed with mean 
and standard deviation 0 and 5 on a cold day and mean and standard 
deviation 15 and 10 on a hot day.
"""

#今日の天気から明日の天気を予想する。隠れマルコフモデル
#昨日以降のデータは関係ない。

#ショートカット作成
tfd = tfp.distributions

#モデル設定
initial_distribution = tfd.Categorical(probs=[0.2, 0.8]) #初日20%で寒くて、80%で暑い
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                 [0.2, 0.8]])  
# 今日が寒い日なら、50%で明日も寒い。今日が暑い日なら20％で明日は寒い。
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[15., 10.])  # refer to point 5 above
#寒い日の分布は平均0度、標準偏差15、暑い日の分布は平均15度、標準偏差10



#モデル適用
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)
# num_steps=7は7日先まで予測するということ


# 場合分けの確立の平均値をとって、明日以降の7日の気温を予測
mean = model.mean()
print(mean.numpy())

