from flask import Flask

from recognition.application import recognition_api

app = Flask(__name__, static_url_path='')

app.register_blueprint(recognition_api)


@app.route('/')
def index():
    return 'hi World'


if __name__ == "__main__":
    # To be removed before deployement 
    # app.debug = True
    app.run()
