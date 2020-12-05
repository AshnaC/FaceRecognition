from flask import Blueprint, request, send_file
# import cv2
# import numpy as np
# from PIL import Image
# import io
from recognition.recognize import recognizeSingleImage

recognition_api = Blueprint('face_recognition_api', __name__)
print('Hi')


# @recognition_api.route('/api/getImage', methods=['POST'])
# def postImage():
#     # print(request.data)
#     file = request.files['file']
#     print(file)
#     npimg = np.fromfile(file, np.uint8)
#     file = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#     file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
#     # img = cv2.imdecode(file)
#     print(file)
#
#     # convert numpy array to PIL Image
#     img = Image.fromarray(file)
#     # create file-object in memory
#     file_object = io.BytesIO()
#     # write PNG in file-object
#     img.save(file_object, 'PNG')
#     # move to beginning of file so `send_file()` it will read from start
#     file_object.seek(0)
#
#     return send_file(file_object, mimetype='image/PNG')

    # data = {'name': 'nabin khadka'}
    # return data


@recognition_api.route('/api/blah', methods=['GET'])
def blah():
    return 'hi World'


@recognition_api.route('/api/getFaces', methods=['POST'])
def getFaces():
    file = request.files['file']
    faces = recognizeSingleImage(file)
    # return faces
    # print(faces)
    data = {'name': 'nabin khadka'}
    return {'result': faces}
