import time
import cv2
import numpy as np  
from PIL import Image, ImageDraw, ImageFont ,ImageColor
import threading




screen_width = 1920
screen_height = 1080


frame_size_mul = 1
frame_width = int(screen_width * frame_size_mul)
frame_height = int(screen_height * frame_size_mul)

# print(f'frame_width:{frame_width}')
# print(f'frame_height:{frame_height}')

# window_size_mul = 0.5
# window_width = int(screen_width * window_size_mul)
# window_height = int(screen_height * window_size_mul)

# print(f'window_width:{window_width}')
# print(f'window_height:{window_height}')


def init_open_camera():
    global cap
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise BaseException
    except BaseException:
        print(f'open camera failed')
    else:

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        # cap.set(cv2.CAP_PROP_FPS,30)
        # cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc('M','J','P','G'))
        # cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc('Y','U','Y','V'))

        

        
        # cv2.namedWindow("Video",cv2.WINDOW_FREERATIO) # 显示图片时：优先铺满，不保持原始图片长宽比
        cv2.namedWindow("Video",cv2.WINDOW_KEEPRATIO)   # 显示图片时：保持原始图片长宽比，优先填充高，宽度不够两边留白
        cv2.setWindowProperty("Video",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print(cap.get(cv2.CAP_PROP_FPS))
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(cap.get(cv2.CAP_PROP_BUFFERSIZE))
        
        

def draw_text_on_image(image, text, position, font_face=cv2.FONT_HERSHEY_COMPLEX, font_scale=1, color=(0, 0, 0), thickness=1):  
    """  
    在图片的特定位置显示文字。  
  
    参数:  
    image -- 输入图片，numpy数组格式  
    text -- 要显示的文字  
    position -- 文字左上角的坐标，格式为(x, y)  
    font_face -- 字体类型（可选，默认为cv2.FONT_HERSHEY_SIMPLEX）  
    font_scale -- 字体大小（可选，默认为1）  
    color -- 文字颜色，格式为(B, G, R)（可选，默认为黑色）  
    thickness -- 文字线条粗细（可选，默认为2）  
  
    返回:  
    带有文字的图片，numpy数组格式  
    """  
    # 获取图片的高度和宽度  
    height, width = image.shape[:2]  
  
    # 确保文字位置在图片范围内  
    if position[0] < 0:  
        position = (0, position[1])  
    if position[1] < 0:  
        position = (position[0], 0)  
    if position[0] + len(text) * font_scale * 10 > width:  
        position = (width - len(text) * font_scale * 10, position[1])  
    if position[1] + font_scale * 20 > height:  
        position = (position[0], height - font_scale * 20)  
  
    # 在图片上添加文字  
    cv2.putText(image, text, position, font_face, font_scale, color, thickness, cv2.LINE_AA)  
  
    # 返回修改后的图片  
    return image  

  
def draw_text_on_image_pil(image, text, position, font, font_size=15, color='red'):  
    """  
    在图片上绘制文本并保存或显示。  
  
    参数:  
    text -- 要绘制的文本  
    position -- 文本左上角坐标，格式为(x, y)  
    font -- 字体 
    font_size -- 字体大小（默认为15）  
    color -- 文本颜色（默认为'red'）  
    """  

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
  



if __name__ == "__main__":

    init_open_camera()
    # 指定字体样式和大小  
    font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'  # 字体文件路径  
    font_style = 'WenQuanYi Zen Hei'  # 你想要使用的字体样式名称  
    font_size = 24  # 字体大小  
    font = ImageFont.truetype(font_path, font_size)  
    
  

    

    
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
            
            # 设置新的图像大小  
            new_size = (1440, 1080)  
            
            # 使用双线性插值调整图像大小  
            # frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LANCZOS4)
            # frame = cv2.resize(frame, new_size)
            
            
            
            # 在图片上添加文字  
            text = "Hello, OpenCV! 你好"  
            position = (50, 100)  # 左上角坐标  
            font_color={255,0,0}
            frame = draw_text_on_image_pil(frame,text,position,font,font_color)
            
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