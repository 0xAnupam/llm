import socket
import ssl
from langchain.llms import CTransformers

# Define the server's address and port
HOST = '127.0.0.1'
PORT = 65432

# Initialize the model
model = CTransformers(model='C:/Users/Anupam/llm/server/llama-2-7b-chat.ggmlv3.q2_K.bin',
                      model_type='llama',
                      config={'max_new_tokens':512,
                              'temperature':0.1})

def handle_client(connection):
    try:
        data = connection.recv(4096).decode('utf-8')
        if data:
            print(f"Received prompt: {data}")
            response = model(data)
            print(response)
            connection.sendall(response.encode('utf-8'))
    except Exception as e:
        print(f"Error handling client: {e}")
    finally:
        connection.close()

def start_server():
    ontext = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = server_socket.accept()
            print(f"Connected by {addr}")
            ssl_conn = context.wrap_socket(conn, server_side=True)
            handle_client(ssl_conn)


if __name__ == "__main__":
    start_server()
