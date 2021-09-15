import os
import pickle
import face_recognition
from sklearn.svm import SVC

path_train = 'celebrity/train'

X_train = []
y_train = []

for cel_name in os.listdir(path_train):
  for img_name in os.listdir(path_train + '/' + cel_name):
    image = face_recognition.load_image_file(path_train + '/' + cel_name + '/' + img_name)
    try:
        face_encoding = face_recognition.face_encodings(image)[0]
        X_train.append(face_encoding)
        y_train.append(cel_name)
    except Exception:
        print('Imagem ' + img_name + ' sem faces')

clf = SVC()
clf = clf.fit(X_train, y_train)

pickle.dump(clf, open('clf.pickle', 'wb'))