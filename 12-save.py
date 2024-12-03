import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import json


# ファイルを開いて読み込む
x_train = np.load('train_X_data.npy')
y_train = np.load('train_Y_data.npy')
x_test = np.load('test_X_data.npy')
y_test = np.load('test_Y_data.npy')

# 正解ラベルを one-hot-encoding にする
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# モデルを作る
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=225))  # input_dim = 15 x 15 = 225
model.add(Dense(10, activation='softmax'))

# モデルをコンパイルする
model.compile(optimizer='rmsprop',
          loss='categorical_crossentropy',
          metrics=['accuracy'])

model.summary()

# 学習する
model.fit(x_train, y_train,
        batch_size=20,
        epochs=30,
        verbose=1)

# モデルの保存
json_string = model.to_json()
open('tegaki-model.json', 'w').write(json_string)

# 重みの保存
hdf5_file = "tegaki-predict.weights.h5"
model.save_weights(hdf5_file)
