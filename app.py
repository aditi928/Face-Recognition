
import json

from flask import Flask, render_template, request, redirect

from face_recognition import face_recognition_api, initialize_test, get_face_list, force_reload_embeddings

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('capture_image.html')


@app.route('/force-reload-users')
def force_reload():
    force_reload_embeddings()
    return redirect('/registered-faces')


@app.route('/upload-user', methods=["POST"])
def upload_user():
    try:
        data = request.files
        if 'file' not in data:
            return {"response": "Invalid Parameters", "data": {}}

        file = data['file']
        file.save('faces/'+file.filename)
        return {"response": "Successfully registered user", "data": {}}
    except Exception as e:
        print(e)
        return {"response": "Unexpected error occurred", "data": {}}


@app.route('/registered-faces')
def registered_faces():
    return render_template('registered-faces.html', faces=get_face_list())


@app.route('/detect-face', methods=["POST"])
def detect_face():
    try:
        data = json.loads(request.get_data())
        if "image" in data:
            image = data["image"].split(",")[1]
        else:
            return {"response": "Invalid Parameters", "data": {}}

    except Exception as e:
        print(e)
        return {"response": "Unexpected error occurred", "data": {}}

    response = face_recognition_api(image)
    return response


@app.route('/detect-custom', methods=["POST"])
def detect_custom():
    try:
        data = request.files
        if 'file' not in data:
            return {"response": "Invalid Parameters", "data": {}}

        image = data['file'].read()
    except Exception as e:
        print(e)
        return {"response": "Unexpected error occurred", "data": {}}

    response = face_recognition_api(image, custom=True)
    return response


initialize_test()


if __name__ == '__main__':
    initialize_test()
    app.run()
