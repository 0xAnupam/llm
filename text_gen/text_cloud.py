from flask import Flask, request, jsonify, render_template
import requests
import time

app = Flask(__name__)

# API configuration
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data['prompt']
    start_time = time.time()
    try:
        output = query({"inputs": prompt})
        print(output)
        end_time = time.time()
        response_time = end_time - start_time
        return jsonify({'response': output, 'response_time': response_time})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
