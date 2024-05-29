import threading  
import time  
  
# 定义一个函数，该函数将被线程执行  
def worker(name,count):  
    """线程执行的函数"""  
    for i in range(count):  
        print(f"{name} is working {i+1}")  
        time.sleep(1)  
  
# 创建线程对象列表  
threads = []  
  
# 创建并启动5个线程  
for i in range(5):  
    t = threading.Thread(target=worker, args=("Alice",5),name="Alice's Thread")  
    threads.append(t)  
    t.start()  
  
# 等待所有线程完成  
for t in threads:  
    t.join()  
  
print("All workers finished.")