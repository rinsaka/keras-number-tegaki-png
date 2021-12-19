from PIL import Image
import numpy as np
import os

filepath = os.path.sep.join(['png', '0-01.png'])
img = Image.open(filepath)
# img = img.convert("P")  ## 白黒に変換
data = np.asarray(img)
print(data)