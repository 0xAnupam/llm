import socket

# Define the server's address and port
HOST = '127.0.0.1'
PORT = 65432

def send_prompt(prompt):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))
        client_socket.sendall(prompt.encode('utf-8'))
        response = client_socket.recv(1024).decode('utf-8')
        print(f"Received response: {response}")

if __name__ == "__main__":
    prompt = input("Enter your prompt: ")
    send_prompt(prompt)
