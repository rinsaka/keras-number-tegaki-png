import glob
import os
from PIL import Image
import numpy as np
import random, math

# png 画像データが保存されているディレクトリの指定
png_dir = "png\\"  #mac では 'png/'
# ファイルを検索
files = glob.glob(png_dir + '*.png')

# 全画像データを格納するためのリストを準備する
png_files = []

for file in files:
    img = Image.open(file)
    img = img.convert("P") # 白黒データに変換
    data = np.asarray(img) # numpy 配列に変換
    cat = file[4]
    data = data.flatten()  # numpy 配列を一次元化
    png_files.append([cat, data]) # リストに追加

# シャッフルして，8割を学習用画像に，2割をテスト用画像にする
random.shuffle(png_files)
threshold = math.floor(len(png_files) * 0.8)
train_data = png_files[0:threshold]
test_data = png_files[threshold:]

print("number of training data : ", len(train_data))
print("number of test data : ", len(test_data))

# 学習用画像
X_train = []
Y_train = []
for train in train_data:
    X_train.append(np.asarray(train[1]))
    Y_train.append(np.asarray(train[0]))

# テスト用画像
X_test = []
Y_test = []
for test in test_data:
    X_test.append(np.asarray(test[1]))
    Y_test.append(np.asarray(test[0]))

XY_train = (X_train, Y_train)

# ファイルに書き出す
np.save('train_X_data.npy', X_train)
np.save('train_Y_data.npy', Y_train)
np.save('test_X_data.npy', X_test)
np.save('test_Y_data.npy', Y_test)

print('ファイルに書き出しました')
