import matplotlib.pyplot as plt
import cv2
from mosaic import mosaic as mosaic

# カスケードファイルを指定する
dir_path = '/Users/s-yajima/Desktop/Python/data/'
cascade_file = 'haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(dir_path + cascade_file)

# 画像を読みこんでグレイスケールに変換する
face_file = 'face3.png'
img = cv2.imread(dir_path + face_file)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# グレイスケールに変換した画像から顔検知を実行し結果を取得する
detect_list = cascade.detectMultiScale(img_gray, minSize=(100, 100))
if len(detect_list) == 0:
    quit()

# 認識した部分に印をつける。検知した顔の数だけ繰り返しモザイク処理を実行する
for(x,y,w,h) in detect_list:
    print('顔の座標 :', x, y, w, h)
    img = mosaic(img, (x, y, x+w, y+h), 10)

# 画像を出力する
output_path = 'face_detect.png'
cv2.imwrite(dir_path + output_path, img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
