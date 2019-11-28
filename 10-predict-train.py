import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

def print_train_predict(idx):
    print('認識結果  ：', np.argmax(pred_train[idx]))
    print('正解ラベル：', np.argmax(y_train[idx]))
    # トレーニングデータの表示
    i = 1
    for x in x_train[idx]:
        if (x == 1):
            print("+ ", end="")
        else:
            print("  ", end="")
        if i % 15 == 0:
            print("")
        i += 1

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

# 認識してみよう
pred_train = model.predict(x_train)


while True:
    print('---------------------')
    print('予測結果を表示したい学習データの番号（0 から', x_train.shape[0]-1 ,'まで）を入力してください（-1で終了します）：', end="")
    str_idx = input()

    # 空の場合の処理
    if str_idx == "":
        print('入力してください')
        continue
    # 入力した文字列を整数に変換するが，変換できない場合のために例外処理が必要
    try:
        idx = int(str_idx)
    except ValueError:
        print('エラー：数字以外の文字は入力できません')
        continue
    # 終了判定
    if idx == -1:
        break
    if idx < 0 or idx > x_train.shape[0]-1:
        print('正しい値を入れてください')
        continue
    print_train_predict(idx)

print('------ 終了しました ------')
