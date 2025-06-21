import os
import json
import base64
import io
import uuid
import pandas as pd
from flask import Flask, request, jsonify, _app_ctx_stack # _app_ctx_stack is deprecated but functional
from flask_cors import CORS
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore, auth
import functools
from datetime import datetime, timedelta

# --- Configuration ---
# IMPORTANT: All sensitive keys are loaded from environment variables for production.
# For Render deployment, ensure these are set in your service's environment settings.
# For local development, set these in your terminal session or use a .env file (not included).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Firebase service account key should be Base64 encoded in the environment variable
FIREBASE_SERVICE_ACCOUNT_KEY_JSON_B64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_B64")

# Define a path for local Firebase credentials file (relative to app.py)
# This file MUST be in your .gitignore! Example: 'firebase-service-account.json'
LOCAL_FIREBASE_CREDENTIALS_FILE = 'firebase-service-account.json'


# --- Flask App Initialization ---
app = Flask(__name__)

# --- CORS Configuration ---
# In production, REPLACE '*' with your FlutterFlow app's deployed domain(s) for security.
# Example: ['https://your-flutterflow-app-domain.com']
# During local development, you might keep '*' or add your local FlutterFlow preview URL.
CORS(app, resources={r"/*": {"origins": "*"}}) 

# --- Initialize Firebase Admin SDK ---
# Preference: Base64 encoded ENV_VAR > Local file
cred = None
cred_load_source = "None" # For logging success message

if FIREBASE_SERVICE_ACCOUNT_KEY_JSON_B64:
    try:
        # Decode the Base64 string from environment variable, then parse as JSON
        decoded_json_string = base64.b64decode(FIREBASE_SERVICE_ACCOUNT_KEY_JSON_B64).decode('utf-8')
        parsed_key_json = json.loads(decoded_json_string)
        
        # The firebase_admin SDK correctly handles '\n' in private_key when loaded from JSON,
        # so explicit .replace('\\n', '\n') is usually not needed here if the original Base64
        # encoding was done from a valid JSON file.
        
        cred = credentials.Certificate(parsed_key_json)
        cred_load_source = "ENV_VAR (Base64 decoded)"
    except Exception as e:
        print(f"ERROR: Failed to prepare Firebase credentials from Base64 ENV_VAR: {e}")
        # If env var fails, proceed to check local file
        cred = None 
elif os.path.exists(LOCAL_FIREBASE_CREDENTIALS_FILE):
    # Local Development Fallback: Load from local file if ENV_VAR not set or failed
    try:
        cred = credentials.Certificate(LOCAL_FIREBASE_CREDENTIALS_FILE)
        cred_load_source = f"local file ({LOCAL_FIREBASE_CREDENTIALS_FILE})"
    except Exception as e:
        print(f"ERROR: Failed to prepare Firebase credentials from local file: {e}")
        cred = None

# Attempt to initialize Firebase app if credentials were successfully loaded
if cred:
    try:
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print(f"Firebase Admin SDK initialized successfully (from {cred_load_source}).")
    except Exception as e:
        print(f"ERROR: Failed to initialize Firebase Admin SDK with prepared credentials. Full trace: {e}")
        db = None
else:
    print("WARNING: Firebase credentials not found or invalid. Firebase operations will fail.")
    db = None

# --- Gemini Model Configuration ---
gemini_model = None # Initialize to None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash') # Using gemini-2.0-flash for speed
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"ERROR: Failed to configure Gemini API. Check GEMINI_API_KEY: {e}")
else:
    print("WARNING: GEMINI_API_KEY environment variable not set. Gemini API calls will fail.")


# --- Authentication Decorator ---
def authenticate_user(f):
    """
    Decorator to verify Firebase ID tokens.
    Adds 'user_id' to Flask's g object if authentication is successful.
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        # Allow health check without authentication
        if request.path == '/health':
            return f(*args, **kwargs)

        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "Authorization header missing"}), 401

        # Expected format: "Bearer <token>"
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return jsonify({"error": "Authorization header must be 'Bearer <token>'"}), 401

        id_token = parts[1]
        try:
            # Verify the ID token against Firebase Auth
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token['uid']
            _app_ctx_stack.top.user_id = uid # Store user ID in application context for endpoint access
            return f(*args, **kwargs)
        except Exception as e:
            print(f"Authentication error: {e}")
            return jsonify({"error": f"Unauthorized: {str(e)}"}), 403
    return decorated_function


# --- Helper Functions for Data Processing & Gemini Calls ---

def parse_data_to_dataframe(file_type: str, file_content_b64: str) -> pd.DataFrame:
    """
    Decodes base64 content and parses it into a Pandas DataFrame based on file_type.
    Supports CSV, JSON, and plain text.
    Raises ValueError for unsupported types or parsing failures.
    """
    try:
        decoded_content = base64.b64decode(file_content_b64).decode('utf-8')
        data_io = io.StringIO(decoded_content)

        if file_type == 'csv':
            df = pd.read_csv(data_io)
        elif file_type == 'json':
            df = pd.read_json(data_io)
        elif file_type == 'text':
            # For plain text, create a simple DataFrame with a single 'content' column
            df = pd.DataFrame({'content': [decoded_content]})
        else:
            raise ValueError(f"Unsupported file type: {file_type}. Must be 'csv', 'json', or 'text'.")

        # Basic data cleaning: drop rows/columns with all NaNs
        df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
        # Fill any remaining NaNs for consistent AI input
        df = df.fillna('')
        return df
    except Exception as e:
        print(f"Error parsing data: {e}")
        raise ValueError(f"Failed to parse data of type {file_type}: {str(e)}. Details: {e}")

def dataframe_to_gemini_context(df: pd.DataFrame) -> str:
    """
    Converts a Pandas DataFrame into a readable string format for Gemini.
    Prioritizes JSON records for structured data, falls back to CSV string.
    Truncates large DataFrames to avoid exceeding Gemini's context window.
    """
    MAX_ROWS_FOR_CONTEXT = 500 # Adjust this limit as needed
    context_str = ""
    truncated_message = ""

    if len(df) > MAX_ROWS_FOR_CONTEXT:
        limited_df = df.head(MAX_ROWS_FOR_CONTEXT)
        truncated_message = f"\n... (truncated {len(df) - MAX_ROWS_FOR_CONTEXT} rows for brevity. Full dataset available for detailed queries via 'data_id'.)"
    else:
        limited_df = df

    try:
        # Attempt to convert to JSON records for better structure for LLMs
        context_str = limited_df.to_json(orient='records', indent=2)
    except Exception:
        # Fallback to CSV string if JSON conversion fails
        context_str = limited_df.to_csv(index=False)
    
    return context_str + truncated_message


def call_gemini_with_schema(prompt_parts: list, response_schema: dict):
    """
    Calls the Gemini API with specified prompt parts and enforces a JSON response schema.
    Raises Exception if Gemini model is not configured or call fails.
    """
    if gemini_model is None:
        raise Exception("Gemini API not configured. Check GEMINI_API_KEY.")

    try:
        generation_config = genai.GenerationConfig(
            temperature=0.7, # Lower temperature for more deterministic, analytical output
            top_k=40,
            top_p=0.95,
            response_mime_type="application/json",
            response_schema=response_schema
        )
        response = gemini_model.generate_content(
            prompt_parts,
            generation_config=generation_config
        )
        # Gemini's structured output comes as a stringified JSON in response.text
        json_response = json.loads(response.text)
        return json_response
    except Exception as e:
        print(f"Error calling Gemini with schema: {e}")
        # Log Gemini's raw response for debugging if available
        # if hasattr(response, 'text'): print(f"Raw Gemini response: {response.text}")
        raise Exception(f"Gemini API call (with schema) failed: {str(e)}")

def call_gemini_text_only(prompt_parts: list):
    """
    Calls the Gemini API for text-only responses.
    Raises Exception if Gemini model is not configured or call fails.
    """
    if gemini_model is None:
        raise Exception("Gemini API not configured. Check GEMINI_API_KEY.")

    try:
        generation_config = genai.GenerationConfig(
            temperature=0.9, # Higher temperature for more creative/conversational output
            top_k=40,
            top_p=0.95,
        )
        response = gemini_model.generate_content(
            prompt_parts,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini for text-only: {e}")
        raise Exception(f"Gemini API call (text-only) failed: {str(e)}")


# --- API Endpoints ---

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint to verify backend is running.
    Does not require authentication.
    """
    return jsonify({"status": "healthy", "message": "InsightWhiz backend is running!"}), 200

@app.route('/upload_data', methods=['POST'])
@authenticate_user
def upload_data():
    """
    Endpoint to receive raw dataset (CSV, JSON, or text), parse it,
    store a representation in Firestore, and generate initial AI insights and chart data.
    Requires Authorization: Bearer <Firebase_ID_Token> in request headers.
    """
    user_id = _app_ctx_stack.top.user_id # Get user_id from authentication decorator
    print(f"[{datetime.now()}] User {user_id} attempting to upload data.")

    # Pre-check for dependencies
    if not db:
        return jsonify({"error": "Firestore not initialized. Backend configuration issue. Check Firebase credentials."}), 500
    if not gemini_model:
        return jsonify({"error": "Gemini API not configured. Backend configuration issue. Check Gemini API key."}), 500

    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    file_type = data.get('file_type')
    file_content_b64 = data.get('file_content_b64')
    file_name = data.get('file_name', 'Unnamed File')

    if not file_type or not file_content_b64:
        return jsonify({"error": "Missing 'file_type' or 'file_content_b64' in payload."}), 400

    try:
        df = parse_data_to_dataframe(file_type, file_content_b64)
        data_representation_for_gemini = dataframe_to_gemini_context(df)

        # Store data context in Firestore
        # Collection path: /users/{userId}/datasets/{dataId}
        datasets_ref = db.collection('users').document(user_id).collection('datasets')
        new_data_ref = datasets_ref.document() # Let Firestore generate a unique document ID
        
        firestore_data = {
            'file_name': file_name,
            'file_type': file_type,
            'data_context': data_representation_for_gemini, # Store truncated context for Gemini, full for retrieval
            'uploaded_at': firestore.SERVER_TIMESTAMP,
            'user_id': user_id
            # In a real app, you might also store a link to the original raw file in Cloud Storage
        }
        new_data_ref.set(firestore_data)
        data_id = new_data_ref.id
        print(f"[{datetime.now()}] Data uploaded by {user_id} and stored in Firestore with ID: {data_id}")

        # --- Prompt for Initial Business Analysis and Chart Data ---
        analysis_prompt_text = f"""
        You are an expert business analyst, specializing in making data actionable. Perform a comprehensive analysis of the provided dataset.
        
        DATASET (file name: '{file_name}', original format: {file_type}, parsed content preview):
        {data_representation_for_gemini}

        Based on this data, provide:
        1. A concise business summary (2-3 paragraphs).
        2. Key insights and actionable trends.
        3. Potential strengths, weaknesses, opportunities, and threats (SWOT) related to the data.
        4. Practical, actionable recommendations based on your analysis.

        Additionally, extract up to 5 key numerical data points suitable for an interactive chart. These should represent significant trends, comparisons, or distributions within the data (e.g., sales over time, expenses by category, customer growth).
        Return the response as a JSON object with two fields:
        'analysis_text': A single string containing the comprehensive text analysis (summary, insights, SWOT, recommendations).
        'chart_data': An array of objects, where each object has 'category' (string, e.g., 'Q1 Sales', 'Marketing Expense'), 'value' (number), and 'label' (string, for display). If no clear chartable data is found or extractable, return an empty array for 'chart_data'.
        """

        # Define the expected JSON schema for Gemini's response
        chart_data_schema = {
            "type": "OBJECT",
            "properties": {
                "analysis_text": { "type": "STRING", "description": "Comprehensive business analysis." },
                "chart_data": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "category": { "type": "STRING", "description": "Category for the chart (e.g., Quarter, Product Name)" },
                            "value": { "type": "NUMBER", "description": "Numerical value for the chart (e.g., Sales, Profit)" },
                            "label": { "type": "STRING", "description": "A brief label for the chart item" }
                        },
                        "required": ["category", "value", "label"]
                    },
                    "description": "An array of structured data points suitable for charting. Max 5 items."
                }
            },
            "required": ["analysis_text", "chart_data"]
        }

        gemini_parts = [{"text": analysis_prompt_text}]
        gemini_response_json = call_gemini_with_schema(gemini_parts, chart_data_schema)
        print(f"[{datetime.now()}] Gemini analysis for {data_id} completed.")

        return jsonify({
            "data_id": data_id,
            "analysis_text": gemini_response_json.get("analysis_text", "No analysis text generated by AI."),
            "chart_data": gemini_response_json.get("chart_data", []),
            "message": "Data uploaded and analyzed successfully."
        }), 200

    except ValueError as ve:
        print(f"[{datetime.now()}] ValueError during upload_data: {ve}")
        return jsonify({"error": str(ve), "message": "Failed to process uploaded data due to format or content issues."}), 400
    except Exception as e:
        print(f"[{datetime.now()}] Unhandled server error during upload_data: {e}")
        return jsonify({"error": f"Internal server error during analysis: {str(e)}", "message": "An unexpected error occurred while analyzing data."}), 500


@app.route('/ask_question', methods=['POST'])
@authenticate_user
def ask_question():
    """
    Endpoint for conversational AI. Users can ask natural language questions
    about their previously uploaded data.
    Requires Authorization: Bearer <Firebase_ID_Token> in request headers.
    """
    user_id = _app_ctx_stack.top.user_id
    print(f"[{datetime.now()}] User {user_id} asking question for data ID: {request.json.get('data_id')}")

    # Pre-check for dependencies
    if not db:
        return jsonify({"error": "Firestore not initialized. Backend configuration issue. Check Firebase credentials."}), 500
    if not gemini_model:
        return jsonify({"error": "Gemini API not configured. Backend configuration issue. Check Gemini API key."}), 500

    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    data_id = data.get('data_id')
    question = data.get('question')

    if not data_id or not question:
        return jsonify({"error": "Missing 'data_id' or 'question' in payload."}), 400

    try:
        # Retrieve data context from Firestore, ensuring it belongs to the authenticated user
        dataset_doc_ref = db.collection('users').document(user_id).collection('datasets').document(data_id)
        dataset_doc = dataset_doc_ref.get()

        if not dataset_doc.exists:
            print(f"[{datetime.now()}] Data ID {data_id} not found for user {user_id} or unauthorized access attempt.")
            return jsonify({"error": "Data ID not found or unauthorized access. Please re-upload data if session expired."}), 404

        data_context = dataset_doc.to_dict().get('data_context')
        file_name = dataset_doc.to_dict().get('file_name', 'your data')

        if not data_context:
            return jsonify({"error": "Data context is empty for this ID. Please re-upload data."}), 404

        # --- Prompt for Conversational Interface ---
        conversational_prompt_text = f"""
        You are a helpful data analysis assistant. Answer the following question based ONLY on the provided dataset context.
        If the answer cannot be directly derived from the data, state that clearly and politely, for example: "Based on the provided data, I cannot answer that question directly."
        The user is asking a question about their dataset: '{file_name}'.

        DATASET CONTEXT:
        {data_context}

        USER QUESTION: {question}
        """

        gemini_parts = [{"text": conversational_prompt_text}]
        ai_answer = call_gemini_text_only(gemini_parts)
        print(f"[{datetime.now()}] Gemini answer for {data_id} generated.")

        return jsonify({
            "answer": ai_answer,
            "message": "Question answered successfully."
        }), 200

    except Exception as e:
        print(f"[{datetime.now()}] Unhandled server error during ask_question: {e}")
        return jsonify({"error": f"Internal server error during conversational query: {str(e)}", "message": "An unexpected error occurred while answering your question."}), 500


# --- Run the Flask App ---
if __name__ == '__main__':
    # For local development:
    # 1. Ensure Python 3.10 is used (e.g., via virtual environment).
    # 2. Set environment variables in your terminal session BEFORE running:
    #    $env:GEMINI_API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY"
    #    $env:FIREBASE_SERVICE_ACCOUNT_KEY_B64 = "YOUR_BASE64_ENCODED_FIREBASE_JSON_STRING_HERE"
    # 3. Alternatively, place your Firebase service account JSON file as 'firebase-service-account.json'
    #    in the same directory as app.py (and add it to .gitignore).
    # Then run: python app.py
    app.run(host='0.0.0.0', port=os.getenv('PORT', 5000), debug=True)

