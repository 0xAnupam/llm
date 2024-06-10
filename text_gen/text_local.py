from flask import Flask, request, jsonify, render_template
from langchain.llms import CTransformers

app = Flask(__name__)

# Initialize the model
model = CTransformers(model='C:/Users/Anupam/gg/llama-2-7b-chat.ggmlv3.q2_K.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.5})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data['prompt']
    try:
        print(prompt)
        response = model(prompt)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
