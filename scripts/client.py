import socket  
  
def start_tcp_client():  
    HOST = 'localhost'  # 如果服务器在树莓派上运行，使用'localhost'或树莓派的IP地址  
    PORT = 10000        # 与服务器相同的端口号  
  
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  
        s.connect((HOST, PORT))  
        print(f'Connected to {HOST}:{PORT}')  

        while True:
            # 接收数据  
            data = s.recv(1024)  
            print('Received', repr(data))  
  
if __name__ == '__main__':  
    start_tcp_client()