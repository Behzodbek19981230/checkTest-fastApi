from flask import Flask, request, jsonify
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

@app.route('/generate-sheet', methods=['POST'])
@requires_auth
def generate_sheet():
    data = request.json or {}
    num_questions = int(data.get('num_questions', 108))
    sheet_id = str(data.get('sheet_id', '00000001'))
    # Faylga generatsiya qilamiz
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    generate_omr_sheet.generate_omr_sheet(tmp_path, num_questions=num_questions, sheet_id=sheet_id)
    # Sheetni o'qib, base64 qilib qaytaramiz
    with open(tmp_path, 'rb') as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return jsonify({'image': img_b64, 'sheet_id': sheet_id, 'num_questions': num_questions})


@app.route('/generate-sheets', methods=['POST'])
@requires_auth
def generate_sheets():
    data = request.json or {}
    sheets = data.get('sheets', [])
    images = []
    for sheet in sheets:
        num_questions = int(sheet.get('num_questions', 108))
        sheet_id = str(sheet.get('sheet_id', '00000001'))
        # Har bir sheet uchun rasm generatsiya qilamiz
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        generate_omr_sheet.generate_omr_sheet(tmp_path, num_questions=num_questions, sheet_id=sheet_id)
        img = Image.open(tmp_path)
        images.append(img)
    # Barcha rasmlarni bitta uzun rasmga birlashtiramiz
    if not images:
        return jsonify({'error': 'No sheets provided'}), 400
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)
    combined = Image.new('RGB', (max_width, total_height), 'white')
    y_offset = 0
    for img in images:
        combined.paste(img, (0, y_offset))
        y_offset += img.height
    # Base64 qilib qaytaramiz
    buf = BytesIO()
    combined.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return jsonify({'image': img_b64})

@app.route('/analyze', methods=['POST'])
@requires_auth
def analyze():
    data = request.json
    img_b64 = data.get('image')
    if not img_b64:
        return jsonify({'error': 'No image provided'}), 400
    # Base64 dan faylga yozish
    img_bytes = base64.b64decode(img_b64)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name
    # OMR tahlil qilish (fayl yo'li orqali)
    result = analyze_omr_sheet(tmp_path)
    return jsonify(result)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
