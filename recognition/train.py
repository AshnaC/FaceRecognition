from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import Normalizer,LabelEncoder
from sklearn.svm import SVC
import pickle
import os

detector = MTCNN()


def extractFace(file):
    # Loading images
    img = cv2.imread(file)
    if img is None:
        return None
    img_cpy = img.copy()
    # Convert the image to RGB,
    # In case the image has an alpha channel or is black and white.
    img_rgb = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(img_rgb)
    # Assuming one face per image during training
    face = result[0]
    x1, y1, width, height = face['box']
    # Sometimes it returns negative coordinates
    x1, y1 = abs(x1), abs(y1)
    # x2, y2 are opposite coordinates
    x2 = x1 + width
    y2 = y1 + height
    # Deriving pixels for face only
    face_px = img_rgb[y1:y2, x1:x2]
    # Resizing the face image to 160px 160px to be consumed by facenet
    face_px = cv2.resize(face_px, (160, 160))
    return face_px


def getFacesFromDir(directory):
    faces = []
    for file in os.listdir(directory):
        path = f'{directory}/{file}'
        face = extractFace(path)
        if face is not None:
            faces.append(face)
    return faces


def load_dataset(dir_path):
    X = []
    y = []
    for subdir in os.listdir(dir_path):
        # Path of sub directory to read images
        path = f'{dir_path}/{subdir}'
        if not os.path.isdir(path):
            continue
        faces = getFacesFromDir(path)
        labels = [subdir] * len(faces)
        print(f'Loaded {len(faces)} pics of {subdir}')
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)


train_dir = "../data"
X_train, y_train = load_dataset(train_dir)

print('Model loading')
model = load_model('./models/facenet_keras.h5')
# Model takes (160,160,3 )input dims
# Model expects standardized pixel inout
print(model.inputs)
# And outputs 128 dim vector for each image
print(model.outputs)

print('Model loaded')


mean = X_train.mean(axis=(0, 1, 2))
std = X_train.std(axis=(0, 1, 2))


def getImageEmbeddings(x):
    # Standardise pixel values across all channels
    x = (x - mean) / std
    # Convert each image to 128 vector
    pred = model.predict(x)
    return pred


X_train_emb = getImageEmbeddings(X_train)

# Normalise embeddings  - Scaling so that the magnitude of vector is 1
# We use l2 normalisation
nm = Normalizer(norm="l2")
Xtrain = nm.transform(X_train_emb)

# Encode labels for tarining
le = LabelEncoder().fit(y_train)
ytrain = le.transform(y_train)

class_model = SVC(kernel='linear', probability=True)
class_model.fit(Xtrain, ytrain)


with open('./models/svc_model.pkl', 'wb') as fid:
    pickle.dump(class_model, fid)

params ={'mean':mean, 'std': std, 'classes':le.classes_}
with open('./models/params.pkl', 'wb') as fid:
    pickle.dump(params, fid)

