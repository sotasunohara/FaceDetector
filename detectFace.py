import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np

# モデルを読み込む
loaded_model = pickle.load(open('model.sav', 'rb'))

hog = cv2.HOGDescriptor()

people = cv2.imread(".\\people\people.jpg", 0)
detection_result = people
plt.imshow(people, cmap='gray')

# 枠の大きさリスト
heightsize_list = [70, 60]
widthsize_list = [60, 50]
num_of_types = len(heightsize_list)

# 集合写真のサイズを求める
height = people.shape[0]
width = people.shape[1]

# 縦方向の移動量
h_mv = 20
# 横方向の移動量
w_mv = 20

for i in range(num_of_types):
    h = 0
    w = 0
    while(h < height - heightsize_list[i]):
        w = 0
        while(w < width - widthsize_list[i]):
            partial_image = people[
                h:h+heightsize_list[i], w:w+widthsize_list[i]]
            partial_image = cv2.resize(partial_image, dsize=(200, 200))
            partial_hog_feature = hog.compute(partial_image)
            # 顔かどうか判別
            result = loaded_model.predict(
                np.array([partial_hog_feature[:, 0]])
                )
            # 顔と判断したら四角形を表示
            if(result == 1):
                detection_result = cv2.rectangle(
                    detection_result,
                    (w, h), (w + widthsize_list[i], h + heightsize_list[i]),
                    color=(0, 0, 0),
                    thickness=5
                )
            w += w_mv
        h += h_mv
plt.imshow(detection_result, cmap='gray')
plt.show()
