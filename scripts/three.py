import cv2  
from multiprocessing import Process, Queue, Value, Array  
import ctypes  
import numpy as np  
  
# 读取相机的子进程  
def capture_process(capture_queue, stop_flag, camera_number):  
    cap = cv2.VideoCapture(camera_number)  
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    while not stop_flag.value:  
        ret, frame = cap.read()  
        if ret:  
            capture_queue.put(frame)  
  
# 显示图像的子进程  
def display_process(display_queue, stop_flag):  
    cv2.namedWindow('Camera Feed', cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty("Camera Feed",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while not stop_flag.value:  
        if not display_queue.empty():  
            frame = display_queue.get()  
            cv2.imshow('Camera Feed', frame)  
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break  
    cv2.destroyAllWindows()  
  
# 主程序  
def main():  
    # 创建共享变量  
    stop_flag = Value(ctypes.c_bool, False)  
    capture_queue = Queue(maxsize=1)  
    display_queue = Queue(maxsize=1)  
  
    # 创建并启动子进程  
    capture_process_p = Process(target=capture_process, args=(capture_queue, stop_flag, 0))  
    display_process_p = Process(target=display_process, args=(display_queue, stop_flag))  
    capture_process_p.start()  
    display_process_p.start()  
  
    try:  
        while True:  
            if not capture_queue.empty():  
                frame = capture_queue.get()  
                display_queue.put(frame)  
    except KeyboardInterrupt:  
        pass  
    finally:  
        # 停止子进程  
        stop_flag.value = True  
        capture_process_p.join()  
        display_process_p.join()  
  
if __name__ == '__main__':  
    main()