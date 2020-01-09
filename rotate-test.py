#
# 画像を１０度ずつ傾けていき顔が認識されるか確認する
#

import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

# カスケードと画像の読込
dir_path = '/Users/s-yajima/Desktop/Python/data/'
face_file = 'face2.png'
cascade_file = 'haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(dir_path + cascade_file)
img = cv2.imread(dir_path + face_file)

# 顔検出を実行する
def detect_face(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_list = cascade.detectMultiScale(img_gray, minSize=(200, 200))
    # 認識した部分に印をつける
    for (x, y, w, h) in face_list:
        print('顔の座標 : ', x, y, w, h)
        color = (0, 100, 0)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=5)

# 角度ごとに顔が検出されるかを検証する
for i in range(0, 9):
    ang = i * 10
    print('---' + str(ang) + '---')
    img_r = ndimage.rotate(img, ang)
    detect_face(img_r)
    plt.subplot(3, 3, i + 1)
    plt.axis('off')
    plt.title('angle=' + str(ang))
    plt.imshow(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))

plt.show()
