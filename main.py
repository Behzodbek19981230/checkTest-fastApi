from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import base64
import cv2
import numpy as np
from PIL import Image
import tempfile
import generate_omr_sheet
from omr_service import analyze_omr_sheet
USERNAME = 'admin'
PASSWORD = 'secret123'

def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

def authenticate():
    return Response(
        'Authentication required', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated



app = Flask(__name__)
CORS(app)

# Default page for runner message
@app.route('/')
def home():
    return '<h2>OMR API Runner: Server is running and ready to accept requests.</h2>'




@app.route('/analyze', methods=['POST'])
@requires_auth
def analyze():
    data = request.json
    img_b64 = data.get('image')
    if not img_b64:
        return jsonify({'error': 'No image provided'}), 400
    # Base64 dan faylga yozish
    try:
        img_bytes = base64.b64decode(img_b64)
    except Exception:
        return jsonify({'error': 'Invalid base64 image'}), 400
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name
    # OMR tahlil qilish (fayl yo'li orqali)
    result = analyze_omr_sheet(tmp_path)
    return jsonify(result)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
