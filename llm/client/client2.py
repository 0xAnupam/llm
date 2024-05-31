from flask import Flask, request, jsonify, render_template
import socket

app = Flask(__name__)

# Define the server's address and port
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 65432

def send_prompt(prompt):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((SERVER_HOST, SERVER_PORT))
        client_socket.sendall(prompt.encode('utf-8'))
        response = client_socket.recv(4096).decode('utf-8')
        return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data['prompt']
    response = send_prompt(prompt)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
