import os
import json
import base64
import io
import uuid
import pandas as pd
import logging
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore, auth
import functools
from datetime import datetime
import fitz  # PyMuPDF for PDF parsing

logging.basicConfig(level=logging.DEBUG)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_SERVICE_ACCOUNT_KEY_JSON_B64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_B64")
LOCAL_FIREBASE_CREDENTIALS_FILE = 'firebase-service-account.json'

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Firebase Init
cred = None
if FIREBASE_SERVICE_ACCOUNT_KEY_JSON_B64:
    try:
        key_json = base64.b64decode(FIREBASE_SERVICE_ACCOUNT_KEY_JSON_B64).decode('utf-8')
        cred = credentials.Certificate(json.loads(key_json))
    except Exception as e:
        logging.error(f"Base64 Firebase credential load failed: {e}")
elif os.path.exists(LOCAL_FIREBASE_CREDENTIALS_FILE):
    try:
        cred = credentials.Certificate(LOCAL_FIREBASE_CREDENTIALS_FILE)
    except Exception as e:
        logging.error(f"Local Firebase credential load failed: {e}")

try:
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logging.info("Firebase initialized")
except Exception as e:
    logging.error(f"Firebase init failed: {e}")
    db = None

# Gemini Init
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        logging.info("Gemini initialized")
    except Exception as e:
        logging.error(f"Gemini init failed: {e}")

# Middleware: Auth
def authenticate_user(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if request.path == '/health':
            return f(*args, **kwargs)
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "Missing Authorization header"}), 401

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return jsonify({"error": "Invalid Authorization format"}), 401

        try:
            decoded_token = auth.verify_id_token(parts[1])
            g.user_id = decoded_token['uid']
            return f(*args, **kwargs)
        except Exception as e:
            logging.error(f"Auth error: {e}")
            return jsonify({"error": "Unauthorized"}), 403
    return wrapper

# Utils
def safe_truncate_context(text, word_limit=3000):
    words = text.split()
    if len(words) > word_limit:
        return ' '.join(words[:word_limit]) + "\n...(truncated)"
    return text

def parse_pdf_to_text(pdf_b64):
    decoded = base64.b64decode(pdf_b64)
    doc = fitz.open(stream=decoded, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def parse_data(file_type, b64_data):
    if file_type == 'pdf':
        text = parse_pdf_to_text(b64_data)
        df = pd.DataFrame({'content': [text]})
    else:
        content = base64.b64decode(b64_data).decode('utf-8')
        if file_type == 'csv':
            df = pd.read_csv(io.StringIO(content))
        elif file_type == 'json':
            df = pd.read_json(io.StringIO(content))
        elif file_type == 'text':
            df = pd.DataFrame({'content': [content]})
        else:
            raise ValueError("Unsupported file type")
    df = df.dropna(how='all').fillna('')
    return df

def df_to_context(df):
    context = df.head(500).to_json(orient='records', indent=2)
    return safe_truncate_context(context)

# Gemini calls
def call_gemini_text(prompt):
    if not gemini_model:
        raise Exception("Gemini not initialized")
    try:
        response = gemini_model.generate_content(
            [{"text": prompt}],
            generation_config=genai.GenerationConfig(temperature=0.9, top_k=40, top_p=0.95)
        )
        return response.text
    except Exception as e:
        logging.error(f"Gemini call error: {e}")
        raise Exception("Gemini text call failed")

def call_gemini_json(prompt, schema):
    if not gemini_model:
        raise Exception("Gemini not initialized")
    try:
        response = gemini_model.generate_content(
            [{"text": prompt}],
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                response_mime_type="application/json",
                response_schema=schema
            )
        )
        return json.loads(response.text)
    except Exception as e:
        logging.error(f"Gemini schema call error: {e}")
        raise Exception("Gemini structured call failed")

# Routes
@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/upload_data', methods=['POST'])
@authenticate_user
def upload():
    data = request.get_json()
    user_id = g.user_id
    file_type = data.get('file_type')
    file_content = data.get('file_content_b64')
    file_name = data.get('file_name', 'Unnamed')

    try:
        df = parse_data(file_type, file_content)
        context = df_to_context(df)

        # Auto-label datasets with AI
        label_prompt = f"Provide 3 short descriptive labels for this dataset based on the content:\n{context}"
        labels_text = call_gemini_text(label_prompt)
        labels = [label.strip() for label in labels_text.split('\n') if label.strip()][:3]

        doc_ref = db.collection('users').document(user_id).collection('datasets').document()
        doc_ref.set({
            'file_name': file_name,
            'file_type': file_type,
            'data_context': context,
            'labels': labels,
            'uploaded_at': firestore.SERVER_TIMESTAMP
        })

        prompt = f"""
        You're a business analyst. Analyze the dataset named '{file_name}' (type: {file_type}).
        Context:
        {context}

        Return JSON with:
        - 'analysis_text': Summary, insights, SWOT, and recommendations.
        - 'chart_data': Up to 5 chart points with 'category', 'value', and 'label'.
        """
        schema = {
            "type": "OBJECT",
            "properties": {
                "analysis_text": {"type": "STRING"},
                "chart_data": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "category": {"type": "STRING"},
                            "value": {"type": "NUMBER"},
                            "label": {"type": "STRING"}
                        },
                        "required": ["category", "value", "label"]
                    }
                }
            },
            "required": ["analysis_text", "chart_data"]
        }

        ai_result = call_gemini_json(prompt, schema)

        return jsonify({
            "data_id": doc_ref.id,
            "analysis_text": ai_result['analysis_text'],
            "chart_data": ai_result['chart_data'],
            "labels": labels
        })
    except Exception as e:
        logging.error(f"Upload failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/ask_question', methods=['POST'])
@authenticate_user
def ask():
    data = request.get_json()
    user_id = g.user_id
    question = data.get('question')
    data_id = data.get('data_id')

    try:
        doc_ref = db.collection('users').document(user_id).collection('datasets').document(data_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Data not found"}), 404

        context = doc.to_dict().get('data_context', '')

        prompt = f"""
        Based on the dataset:
        {context}

        User's question: {question}

        Answer strictly based on data. If not possible, say so clearly.
        """
        answer = call_gemini_text(prompt)

        # Save Q&A history
        qa_collection = doc_ref.collection('qa_history')
        qa_collection.add({
            'question': question,
            'answer': answer,
            'asked_at': firestore.SERVER_TIMESTAMP
        })

        return jsonify({"answer": answer})
    except Exception as e:
        logging.error(f"Ask failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
