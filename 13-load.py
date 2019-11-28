import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import json
from keras.models import model_from_json

# 保存したモデルの読み込み（10-save-model.py で作成された）
model = model_from_json(open('tegaki-model.json').read())
# 保存した重みの読み込み
model.load_weights('tegaki-predict.hdf5')

model.summary()
