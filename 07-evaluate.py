import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

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

# 学習してみよう（このコードだけで，学習状況も表示される）
model.fit(x_train, y_train,
        batch_size=20,
        epochs=30,
        verbose=1)

# モデルを評価する（テストデータを使う）
score = model.evaluate(x_test, y_test)

# 評価結果を表示する
print(score)
print(model.metrics_names)
print(model.metrics_names[0], " : ", score[0])
print(model.metrics_names[1], " : ", score[1])
