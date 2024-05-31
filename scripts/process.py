import cv2  
import time
import threading  
import numpy as np  

from queue import Queue 
from PIL import Image, ImageDraw, ImageFont ,ImageColor
import socket  

from multiprocessing import Process ,Queue









    
    





    






def draw_text_on_image_pil(image, text, position, font, font_size=15, color='red'):  

    rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)  
  

    if isinstance(color, str):  
        # 如果颜色是字符串（如'red'），则转换为RGB元组  
        color = ImageColor.getrgb(color)  

  
    # 在图片上绘制文本  
    draw.text(position, text, font=font, fill=color)  
    
    
    rgb_image = np.array(pil_image)
    bgr_image = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2BGR)
  
    return bgr_image
  
# 读取相机帧的线程函数  
def read_thread_worker(frame_queue): 
    print("read_thread_worker")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise BaseException
    except BaseException:
        print(f'open camera failed')
    else:
        
        
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)


    print(cap.get(cv2.CAP_PROP_FPS))
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 获取FourCC  
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  
    # 将FourCC转换为对应的字符  
    fourcc_char = chr((fourcc >> 0) & 0xFF) + chr((fourcc >> 8) & 0xFF) + chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)  
    print(f"Current FourCC: {fourcc_char}") 

    
    font = ImageFont.truetype('/usr/share/fonts/truetype/google/NotoSansCJKsc.ttf', 20)
    text = "Hello, OpenCV! 你好,显示图片时：优先铺满，不保持原始图片长宽比,显示图片时：优先铺满，不保持原始图片长宽比,显示图片时：优先铺满，不保持原始图片长宽比,显示图片时：优先铺满，不保持原始图片长宽比" 
    position = (50, 100)  # 左上角坐标  
    font_color={255,0,0}
    
    # 记录开始时间和帧计数  
    start_time = time.time()  
    frame_count = 0 
    
 
    while True: 
        ret, frame = cap.read()  
        if ret:  
            if not frame_queue.full(): 
                # frame = draw_text_on_image_pil(frame,text,position,font,font_color)
                frame_queue.put(frame)
                frame_count+=1
                
                
                
            else:
                # print("full")  
                pass

            
        else:  
            break  

        # 运行一定时间后退出循环，以避免无限循环  
        if (time.time() - start_time) > 20:  # 例如，运行10秒钟  
            break  
    
    # 计算并打印估算的帧率  
    elapsed_time = time.time() - start_time  
    estimated_fps = frame_count / elapsed_time  
    print(f"input FPS: {estimated_fps:.2f} FRAMES:{frame_count}")

# resize frame
def resize_thread_worker():
    global cap,frame_queue,resize_queue
    # 设置新的图像大小  
    new_size = (1440, 1080)    

  

    
    start_time = time.time() 
    frame_count = 0  
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # frame = cv2.resize(frame, new_size,cv2.INTER_LANCZOS4)

            


            if not resize_queue.full():
                resize_queue.put(frame) 
                frame_count+=1
        # 运行一定时间后退出循环，以避免无限循环  
        if (time.time() - start_time) > 20:  # 例如，运行10秒钟  
            break  
    # 计算并打印估算的帧率  
    elapsed_time = time.time() - start_time  
    estimated_fps = frame_count / elapsed_time  
    print(f"resize FPS: {estimated_fps:.2f} FRAMES:{frame_count}")

# puttext_frame
def puttext_thread_worker():
    global cap,frame_queue,puttext_queue
    font = ImageFont.truetype('/usr/share/fonts/truetype/google/NotoSansCJKsc.ttf', 40)
    text = "Hello, OpenCV! 你好,显示图片时：优先铺满，不保持原始图片长宽比,显示图片时：优先铺满，不保持原始图片长宽比,显示图片时：优先铺满，不保持原始图片长宽比,显示图片时：优先铺满，不保持原始图片长宽比"  
    position = (50, 100)  # 左上角坐标  
    font_color={255,0,0}
    
    start_time = time.time() 
    frame_count = 0  
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame = draw_text_on_image_pil(frame,text,position,font,font_color)

            if not puttext_queue.full():
                puttext_queue.put(frame) 
                frame_count+=1
        # 运行一定时间后退出循环，以避免无限循环  
        if (time.time() - start_time) > 20:  # 例如，运行10秒钟  
            break  
    # 计算并打印估算的帧率  
    elapsed_time = time.time() - start_time  
    estimated_fps = frame_count / elapsed_time  
    print(f"puttext FPS: {estimated_fps:.2f} FRAMES:{frame_count}")
  
# 显示队列中帧的线程函数  
def display_thread_worker(frame_queue):  

    print("display_thread_worker")
    
    # cv2.namedWindow("Frame",cv2.WINDOW_FREERATIO) # 显示图片时：优先铺满，不保持原始图片长宽比
    cv2.namedWindow("Frame",cv2.WINDOW_KEEPRATIO)   # 显示图片时：保持原始图片长宽比，优先填充高，宽度不够两边留白
    cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    




    
    
    # 记录开始时间和帧计数  
    start_time = time.time()  
    frame_count = 0  
    empty_count = 0

    while True:  
        if not frame_queue.empty():  # 确保队列中有帧  
            frame = frame_queue.get() 
            cv2.imshow("Frame", frame) 

            
            key=cv2.waitKey(1)
            if key!=-1:
                print('key:',key)


            frame_count += 1 
            # print(f'frame_count out:{frame_count}') 
    
            
        else:
            # time.sleep(0.01)
            # empty_count+=1
            # print(f'empty:{empty_count}')
            pass
            
        # 运行一定时间后退出循环，以避免无限循环  
        if (time.time() - start_time) > 20:  # 例如，运行10秒钟  
            break  


    # 计算并打印估算的帧率  
    elapsed_time = time.time() - start_time  
    estimated_fps = frame_count / elapsed_time  
    print(f"out FPS: {estimated_fps:.2f} FRAMES:{frame_count}")
  



    



        

  
if __name__ == "__main__":
    



    frame_queue = Queue(maxsize=10)
    resize_queue = Queue(maxsize=2)
    puttext_queue = Queue(maxsize=2)
    




    read_process = Process(target=read_thread_worker,args=(frame_queue,))
    read_process.start()
    
    display_process = Process(target=display_thread_worker,args=(frame_queue,))
    display_process.start()
    
    read_process.join()
    display_process.join()
    
    print('join')
  