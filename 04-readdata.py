import numpy as np
from keras.utils.np_utils import to_categorical

# ファイルを開いて読み込む
x_train = np.load('train_X_data.npy')
y_train = np.load('train_Y_data.npy')
x_test = np.load('test_X_data.npy')
y_test = np.load('test_Y_data.npy')

print(y_train)
# 正解ラベルを one-hot-encoding にする
y_train = to_categorical(y_train, 10)
print(y_train)
print(y_test)
# 正解ラベルを one-hot-encoding にする
y_test = to_categorical(y_test, 10)
print(y_test)
