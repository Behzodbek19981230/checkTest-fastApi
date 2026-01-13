from flask import Flask, request, jsonify, Response, g
from flask_cors import CORS
from functools import wraps
import logging
import os
import traceback
import uuid
import base64
import cv2
import numpy as np
from PIL import Image
import tempfile
import generate_omr_sheet
from omr_service import analyze_omr_sheet
from import_service import parse_docx_questions, parse_xlsx_questions
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

# Ensure Uzbek text is returned correctly
app.config['JSON_AS_ASCII'] = False

# Basic structured logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO').upper())


@app.before_request
def _assign_request_id():
    g.request_id = request.headers.get('X-Request-Id') or str(uuid.uuid4())


@app.errorhandler(Exception)
def _handle_unexpected_error(e):
    """Return JSON on unhandled errors (prevents default HTML 500 pages)."""
    request_id = getattr(g, 'request_id', None) or str(uuid.uuid4())
    app.logger.exception('Unhandled error (requestId=%s)', request_id)
    payload = {
        'message': 'Internal Server Error',
        'requestId': request_id,
        'error': str(e),
        'errorType': type(e).__name__,
    }
    # Only include traceback when explicitly enabled (avoid leaking internals by default)
    if str(os.getenv('DEBUG_ERRORS', '')).strip().lower() in {'1', 'true', 'yes', 'on'}:
        payload['traceback'] = traceback.format_exc()
    return jsonify(payload), 500

PARSER_VERSION = '2026-01-12-img-dedupe'

# Default page for runner message
@app.route('/')
def home():
    return '<h2>OMR API Runner: Server is running and ready to accept requests.</h2>'


@app.route('/version')
def version():
	return jsonify({'parserVersion': PARSER_VERSION})




@app.route('/analyze', methods=['POST'])
@requires_auth
def analyze():
    data = request.json
    img_b64 = data.get('image')
    code_digits = data.get('codeDigits', 5) if isinstance(data, dict) else 5
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
    try:
        code_digits_int = int(code_digits)
    except Exception:
        return jsonify({'error': 'codeDigits must be an integer'}), 400

    result = analyze_omr_sheet(tmp_path, code_digits=code_digits_int)
    return jsonify(result)


@app.route('/import/questions', methods=['POST'])
def import_questions():
    """Import questions from DOCX/XLSX.

    Expects multipart/form-data with a `file` field.
    Returns: { questions: [{question, options, correctAnswer, points}], errors: [..] }
    """
    request_id = getattr(g, 'request_id', None) or str(uuid.uuid4())
    f = request.files.get('file')
    if not f:
        return jsonify({'message': 'No file uploaded (field name must be "file")', 'requestId': request_id}), 400

    filename = (f.filename or '').lower()
    content = f.read()

    try:
        if filename.endswith('.docx'):
            parsed, errors = parse_docx_questions(content)
        elif filename.endswith('.xlsx') or filename.endswith('.xlsm') or filename.endswith('.xltx') or filename.endswith('.xltm'):
            parsed, errors = parse_xlsx_questions(content)
        else:
            return jsonify({'message': 'Unsupported file type. Upload .docx or .xlsx', 'requestId': request_id}), 415
    except Exception as e:
        app.logger.exception('Failed to parse import file (requestId=%s, filename=%s)', request_id, filename)
        return jsonify({'message': 'Failed to parse file', 'error': str(e), 'requestId': request_id}), 500

    # Be defensive about JSON serialization.
    questions_out = []
    for q in parsed:
        questions_out.append({
            'question': str(q.question),
            'options': [str(o) for o in (q.options or [])],
            'correctAnswer': int(q.correct_answer_index),
            'points': int(q.points),
        })

    app.logger.info('Import parsed %s questions (%s errors) requestId=%s file=%s', len(parsed), len(errors), request_id, filename)

    return jsonify({
        'parserVersion': PARSER_VERSION,
        'requestId': request_id,
        'questions': questions_out,
        'errors': errors,
    })




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
