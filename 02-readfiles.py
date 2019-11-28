import glob
import os
from PIL import Image
import numpy as np

# png 画像データが保存されているディレクトリの指定
png_dir = 'png\\'  #mac では 'png/'
# ファイルを検索
files = glob.glob(png_dir + '*.png')
# print(files)

# 全画像データを格納するためのリストを準備する
png_files = []

for file in files:
    img = Image.open(file)
    img = img.convert("P") # 白黒データに変換
    data = np.asarray(img) # numpy 配列に変換
    cat = file[4]
    data = data.flatten()  # numpy 配列を一次元化
    png_files.append([cat, data]) # リストに追加

print(png_files[81])  # 数字(0-99)を適当に変えて試すと良い
