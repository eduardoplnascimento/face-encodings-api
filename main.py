import pickle
import face_recognition
import json
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encoding', methods=['POST'])
def encoding():
    try:
        file = request.files['file']
        image = face_recognition.load_image_file(file)
        face_encoding = face_recognition.face_encodings(image)[0]
    except Exception:
        return json.dumps({'success': False, 'encodings': None})
    return json.dumps({'success': True, 'encodings': face_encoding.tolist()})

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        image = face_recognition.load_image_file(file)
        face_encoding = face_recognition.face_encodings(image)[0]
        clf = pickle.load(open('clf.pickle', 'rb'))
        predict = clf.predict([face_encoding])[0]
    except Exception:
        predict = 'Sem faces'
    return render_template('predict.html', predict=predict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)