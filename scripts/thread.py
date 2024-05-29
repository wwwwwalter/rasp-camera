import threading  
import time  
import cv2
import numpy as np  
from PIL import Image, ImageDraw, ImageFont ,ImageColor


def init_open_camera():
    global cap
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise BaseException
    except BaseException:
        print(f'open camera failed')
    else:

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
        cap.set(cv2.CAP_PROP_FPS,30)
        

        
        # cv2.namedWindow("Video",cv2.WINDOW_FREERATIO) # 显示图片时：优先铺满，不保持原始图片长宽比
        cv2.namedWindow("Video",cv2.WINDOW_KEEPRATIO)   # 显示图片时：保持原始图片长宽比，有限填充高，宽度不够两边留白
        # cv2.setWindowProperty("Video",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print(cap.get(cv2.CAP_PROP_FPS))
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(cap.get(cv2.CAP_PROP_BUFFERSIZE))
  
  
def capture():  
    init_open_camera()
    # 记录开始时间和帧计数  
    start_time = time.time()  
    frame_count = 0  
    
    # 循环读取摄像头的帧  
    try:  
        while True:  
            ret, frame = cap.read()  
    
            if not ret:  
                print("Error: Can't receive frame (stream end?). Exiting ...")  
                break  
    
            # 你可以在这里处理或显示帧  
            
            
            
            # 显示图片     
            cv2.imshow('Video', frame)  
            
            
            
             # 等待按键，如果按下'q'键则退出循环  
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break  
            
            frame_count += 1  
    
            # 运行一定时间后退出循环，以避免无限循环  
            if (time.time() - start_time) > 10:  # 例如，运行10秒钟  
                break  
    
    except KeyboardInterrupt:  
        # 允许使用Ctrl+C退出循环  
        pass  
    
    finally:  
        # 释放摄像头资源  
        cap.release()  
        cv2.destroyAllWindows()  
    
        # 计算并打印估算的帧率  
        elapsed_time = time.time() - start_time  
        estimated_fps = frame_count / elapsed_time  
        print(f"Estimated FPS: {estimated_fps:.2f}")
    
def putText():
    # 获取帧的高度和宽度  
    height=480
    width=640
    
        
    # 创建一个与帧大小相同的透明图层（BGR）  
    overlay = np.zeros((height, width, 3), dtype=np.uint8)  
    
    # 在透明图层上添加文字信息  
    # 注意：OpenCV不支持真正的透明文本，所以我们只是降低文字颜色的亮度来模拟透明效果  
    font = cv2.FONT_HERSHEY_SIMPLEX  
    fontScale = 1  
    fontColor = (0, 255, 0)  # BGR颜色，这里使用绿色  
    thickness = 2  
    text = "Hello, OpenCV!"  
    textSize, baseline = cv2.getTextSize(text, font, fontScale, thickness)  
    textX = 10  
    textY = height - 10 - baseline  
    cv2.putText(overlay, text, (textX, textY), font, fontScale, fontColor, thickness)  
    
    cv2.namedWindow("text",cv2.WINDOW_KEEPRATIO)
    while True:
        
        cv2.imshow("text",overlay)
        # 等待按键，如果按下'q'键则退出循环  
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break  
    
          
  
# 创建一个Thread实例  
t1 = threading.Thread(target=capture)
t2 = threading.Thread(target=putText)  
  
# 启动线程  
t2.start()
time.sleep(3)
t1.start()  
  
t1.join()
t2.join()

  
# 打印线程状态  
print("All workers finished.")