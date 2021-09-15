import json
import face_recognition
from flask import Flask, request

app = Flask(__name__)

@app.route('/encoding', methods=['POST'])
def encoding():
    try:
        file = request.files['file']
        image = face_recognition.load_image_file(file)
        face_encoding = face_recognition.face_encodings(image)[0]
    except Exception:
        return json.dumps({'success': False, 'encodings': None})
    return json.dumps({'success': True, 'encodings': face_encoding.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)