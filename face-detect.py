#
# OpenCVにより画像から顔を検出する
#
import matplotlib.pyplot as plt
import cv2

# カスケードファイルを指定して検出器を作成する
dir_path = '/Users/s-yajima/Desktop/Python/data/'
cascade_file = 'haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(dir_path + cascade_file)

# 画像を読みこんでグレイスケールに変換する
#face_file = 'face.jpg'
face_file = 'face2.png'
img = cv2.imread(dir_path + face_file)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 顔検知を実行し結果を取得する
detect_list = cascade.detectMultiScale(img_gray, minSize=(150, 150))

# 結果を確認
if len(detect_list) == 0:
    print('顔認識に失敗しました.')
    quit()

# 認識した部分に印をつける
for(x,y,w,h) in detect_list:
    print('顔の座標 :', x, y, w, h)
    color = (0, 200, 0)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=5)

# 画像から最後に印をつけた顔の部分を切り取り別の画像として保存する
cut_file = 'cut_face.jpg'
img2 = img[y:(y+h), x:(x+w)]
cv2.imwrite(dir_path + cut_file, img2)

# 画像を出力する
output_path = 'face_detect.png'
cv2.imwrite(dir_path + output_path, img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

