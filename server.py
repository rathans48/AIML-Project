import socket
import sys

HOST = '127.0.0.1' 
PORT = 8080

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"Server started. Go to http://{HOST}:{PORT} in your browser.")
        print("Waiting for request...")

        while True:
            conn, addr = s.accept()
            with conn:
                print("Got a request!")
                request = conn.recv(1024)
                response = b"HTTP/1.1 200 OK\n\nGot a request!"
                conn.sendall(response)
except Exception as e:
    print(f"Error: {e}")