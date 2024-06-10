from flask import Flask, request, jsonify, render_template, send_file
import requests
import time
import io
from PIL import Image

app = Flask(__name__)

# API configuration
API_URL = "https://api-inference.huggingface.co/models/Melonie/text_to_image_finetuned"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    return response.content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data['prompt']
    start_time = time.time()
    try:
        image_bytes = query({"inputs": prompt})
        end_time = time.time()
        response_time = end_time - start_time
        
        # Save the image to a BytesIO object
        image = Image.open(io.BytesIO(image_bytes))
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png', as_attachment=False, download_name='generated.png'), 200, {'Response-Time': str(response_time)}
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
