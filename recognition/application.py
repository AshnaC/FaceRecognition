from flask import Blueprint, request, send_file
from recognition.recognize import recognizeSingleImage

recognition_api = Blueprint('face_recognition_api', __name__)

@recognition_api.route('/api/getFaces', methods=['POST'])
def getFaces():
    file = request.files['file']
    faces = recognizeSingleImage(file)
    return {'result': faces}
