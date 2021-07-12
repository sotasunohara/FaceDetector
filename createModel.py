import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pandas as pd
import pickle

face_path = '.\\face\\*'
notface_path = '.\\noface\\*'

hog = cv2.HOGDescriptor()


# HOG特徴量を求める
def getHOGFeature(filePath):
    files = glob.glob(filePath)
    hogfeature = []
    for filename in files:
        image = cv2.imread(filename, 0)
        image = cv2.resize(image, dsize=(200, 200))
        hogfeature.append(hog.compute(image))
    return np.array(hogfeature)


# HOG特徴量を抽出
print('HOG特徴量抽出中')
face_hogfeature = getHOGFeature(face_path)
notface_hogfeature = getHOGFeature(notface_path)

X = np.concatenate([face_hogfeature, notface_hogfeature])[:, :, 0]
y_face = np.ones(face_hogfeature.shape[0])
y_notface = np.zeros(notface_hogfeature.shape[0])
y = np.concatenate([y_face, y_notface])

df = pd.DataFrame(X)
df['isFace'] = y

# データをシャッフルする
df_shuffled = df.sample(frac=1).reset_index(drop=True)


# Numpy配列にする
X_shuffled = df_shuffled.drop('isFace', axis=1).to_numpy()
y_shuffled = df_shuffled['isFace'].to_numpy()

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X_shuffled, y_shuffled, test_size=0.3)

# SVM
print('学習中')
model = LinearSVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# スコア
print(model.score(X_test, y_test))

model_filename = 'model.sav'
pickle.dump(model, open(model_filename, 'wb'))
