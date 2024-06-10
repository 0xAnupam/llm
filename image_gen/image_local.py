from flask import Flask, request, jsonify, render_template, send_file
import torch
from torchvision import transforms as tfms
from PIL import Image
from diffusers import StableDiffusionPipeline
import io
import time

app = Flask(__name__)

# Load the Stable Diffusion model
model_path = "sd-v1-4.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data['prompt']
    start_time = time.time()
    
    with torch.no_grad():
        image = pipe(prompt, guidance_scale=7.5)["sample"][0]
    
    end_time = time.time()
    response_time = end_time - start_time

    # Save the image to a BytesIO object
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png', as_attachment=False, download_name='generated.png'), 200, {'Response-Time': str(response_time)}

if __name__ == "__main__":
    app.run(debug=True)
