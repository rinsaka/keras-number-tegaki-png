# 手書き数字の認識
- データの作成から始める
- https://rinsaka.com/python/tegaki-number-index.html
- https://github.com/rinsaka/keras-number-tegaki-png

### トレーニングデータ
- png/ ディレクトリにある
- 15 x 15 ピクセルの手書きイメージ

### 01-readfile.py
- 一つのファイルを開いて，白黒に変換して表示する
- Mac, Linux の場合は，フォルダの指定方法を変更する必要あり

### 02-readfiles.py
- 画像データの一覧を読み込んでみよう
- Mac, Linux の場合は，フォルダの指定方法を変更する必要あり

### 03-data.py
- 学習データとテストデータを準備する
- Mac, Linux の場合は，フォルダの指定方法を変更する必要あり

### 04-readdata.py
- 保存したデータを開いてみる

### 05-model.py
- モデルを作る

### 06-train.py
- 学習（トレーニング）させてみよう

### 07-evaluate.py
- モデルを評価しよう

### 08-predict-train.py
- 学習データで認識させてみよう(1)

### 09-predict-train.py
- 学習データで認識させてみよう(2)

### 10-predict-train.py
- 学習データで認識させてみよう(3)

### 11-predict-test.py
- テストデータで認識させてみよう

### 12-save.py
- モデルと重みパラメータを保存しよう

### 13-load.py
- 学習済みモデルをロードしよう

### 14-load-predict.py
- 学習済みモデルをロードして，認識してみよう
