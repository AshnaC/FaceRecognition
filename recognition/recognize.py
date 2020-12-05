from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import Normalizer
import pickle
from PIL import Image
import io
from base64 import encodebytes

detector = MTCNN()

with open('recognition/models/svc_model.pkl', 'rb') as fid:
    svc_model = pickle.load(fid)
with open('recognition/models/params.pkl', 'rb') as fid:
    image_params = pickle.load(fid)


def extractFaces(x):
    # Loading images
    # img = cv2.imread(file)
    # img_cpy = img.copy()
    # Convert the image to RGB,
    # In case the image has an alpha channel or is black and white.
    npimg = np.fromfile(x, np.uint8)
    file = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)

    img_rgb = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(img_rgb)
    face_imgs = []
    for face in result:
        x1, y1, width, height = face['box']
        # Sometimes it returns negative coordinates
        x1, y1 = abs(x1), abs(y1)
        # x2, y2 are opposite coordinates
        x2 = x1 + width
        y2 = y1 + height
        face_px = img_rgb[y1:y2, x1:x2]
        # Resizing the face image to 160px 160px to be consumed by facenet
        face_px = cv2.resize(face_px, (160, 160))
        face_px = cv2.cvtColor(face_px, cv2.COLOR_BGR2RGB)
        face_imgs.append(face_px)
    return np.asarray(face_imgs)


model = load_model('recognition/models/facenet_keras.h5')
mean = image_params['mean']
std = image_params['std']
classes = image_params['classes']


def getImageEmbeddings(x):
    # Standardise pixel values across all channels
    x = (x - mean) / std
    # Convert each image to 128 vector
    pred = model.predict(x)
    return pred


nm = Normalizer(norm="l2")


def normaliseEmbeddings(x):
    # Normalise embeddings
    return nm.transform(x)


def recognizeSingleImage(file_name):
    faces_px = extractFaces(file_name)
    X = getImageEmbeddings(faces_px)
    X = normaliseEmbeddings(X)
    pred_label = svc_model.predict(X)
    pred_prob = np.amax(svc_model.predict_proba(X), axis=1)
    #     img = cv2.imread(file_name)
    faces = []
    print('People',pred_prob)
    for i, prob in enumerate(pred_prob):
        if prob > 0.56:
            print(classes[pred_label[i]], prob)
            img = Image.fromarray(faces_px[i])
            # create file-object in memory
            file_object = io.BytesIO()
            # write PNG in file-object
            img.save(file_object, 'PNG')
            encoded_img = encodebytes(file_object.getvalue()).decode('ascii')
            faces.append({'img': encoded_img, 'label': classes[pred_label[i]]})
    return faces
