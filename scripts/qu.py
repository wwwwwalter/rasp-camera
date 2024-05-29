import cv2  
import time
import threading  
import numpy as np  
import RPi.GPIO as GPIO  
from queue import Queue 
from PIL import Image, ImageDraw, ImageFont ,ImageColor
import socket  

  
# 设置GPIO模式为BCM  
GPIO.setmode(GPIO.BCM)  
  
# 设置红外接收器的GPIO引脚（这里以18为例，请根据实际情况修改）  
IR_PIN = 18  
GPIO.setup(IR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  



# 遥控器按键字典
remote_controller_mapping = {
    '01000101':'1',
    '01000110':'2',
    '01000111':'3',
    '01000100':'4',
    '01000000':'5',
    '01000011':'6',
    '00000111':'7',
    '00010101':'8',
    '00001001':'9',
    '00011001':'0',

    '00010110':'*',
    '00001101':'#',

    '00011000':"up",
    '01010010':"down",
    '00001000':"left",
    '01011010':"right",
    '00011100':"ok",
}





# 校验地址码或者数据码
def check_code(code,code_inversion):
    inverted_str = ''  
    for char in code:  
        if char == '0':  
            inverted_str += '1'  
        elif char == '1':  
            inverted_str += '0'  
    if inverted_str == code_inversion:
        return True
    else:
        return False

# 红外NEC协议
# https://www.cnblogs.com/ivan0512/p/15312394.html
# 红外NEC协议(协议定义和实际接收器针脚电平相反)
# 引导码|地址码|地址码反码|数据码|数据码反码|停止位
# 引导码：9ms(高)+
    # 4.5ms(低) 表示当前帧是数据帧
    # 2.5ms(低) 表示当前帧是重复帧
# 地址码，地址码反码，数据码，数据码反码：8bit*4[LSB小端]优先传输低字节
# 逻辑1：0.56ms(高)+1.69ms(低)=2.25ms
# 逻辑0：0.56ms(高)+0.56ms(低)=1.12ms
# 停止位：0.56ms(低)



def parse_keyvalue(code):
    # 去除包含'-'的帧
    if '-' in code:
        return
    
    
    # 解析（左拼接）
    # 例帧：10111010 01000101 11111111 00000000
    #       数据反码  数据码   地址反码  地址码
    data_inversion = code[0:8]
    data_code = code[8:16]
    address_inversion = code[16:24]
    address_code = code[24:32]
    
    print(f':{data_code}')
    print(f':{data_inversion}')
    print(f':{address_code}')
    print(f':{address_inversion}')
    
    # 校验地址
    inverted_str=''
    for char in address_code:
        if char == '0':
            inverted_str+='1'
        elif char == '1':
            inverted_str+='0'
    if inverted_str != address_inversion:
        print(f'地址有误')
        # return
    
    # 校验键值
    inverted_str=''
    for char in data_code:
        if char == '0':
            inverted_str+='1'
        elif char == '1':
            inverted_str+='0'
    if inverted_str != data_inversion:
        print(f'键值有误')
        return
    
    
    print('校验通过')
    device = address_code
    key = remote_controller_mapping.get(data_code)
    print(f'Id = {device} Key = {key}')
        
    
    



rising_index = 0
rising_edge_time=0
error_flag=False
keycode=""
# 定义一个回调函数，当引脚状态变化时调用 
def rising(channel):
    global rising_index,rising_edge_time,keycode
    rising_index+=1
    
    last_time = rising_edge_time
    if last_time != 0:
        if (time.time()-last_time)*1000>150:
            rising_index=1
            print('归一')
    
    
    
    # 头脉冲
    if(rising_index == 1):
        keycode=""
        rising_edge_time = time.time()
        print(f'[{rising_index:02d}]')
    
    # bit 1
    # 根据bit 1和head来判断当前帧是数据帧还是重复帧
    elif rising_index == 2:
        last_time = rising_edge_time
        rising_edge_time = time.time()
        head_time = (rising_edge_time-last_time)*1000

        # 数据帧
        if 4<head_time<5.5:
            print(f'[{rising_index:02d}]数据帧:{head_time}')
            
        # 重复帧
        elif 2.5<head_time<3:
            print(f'[{rising_index:02d}]重复帧:{head_time}')
            rising_index=0
        # 错误信号
        else:
            print(f'[{rising_index:02d}]错误帧:{head_time}')
            rising_index=0
    

        
            
            
        


    # 通过前两个脉冲检查无误后，解析(3~34)1~32 bit
    else:
        last_time = rising_edge_time
        rising_edge_time = time.time()
        data_time = (rising_edge_time-last_time)*1000
        
        if 1.7<data_time<5:
            keycode='1' + keycode
            print(f'[{rising_index-2:02d}][1]:{data_time}')
        elif 0<data_time<1.7:
            keycode='0' + keycode
            print(f'[{rising_index-2:02d}][0]:{data_time}')
        else:
            keycode='-' + keycode
            print(f'[{rising_index-2:02d}][-]:{data_time}')
            pass
        if rising_index == 34:
            rising_index = 0 
            print(f'keycode:{keycode}')
            # 校验数据并解析数据帧
            parse_keyvalue(keycode)
            
        if (rising_index-2)%8 == 0:
            print('\n')



    


def control_thread_worker():
    global key_value
    # 记录开始时间和帧计数  
    start_time = time.time()  
    frame_count = 0 
    try:  
        while True: 
            if key_value == "#":
                # key_value=""
                break 
            if GPIO.input(IR_PIN)==0:
                try:
                    address_code,data_code = ir_receiver_nec_decode(IR_PIN)
                    if address_code != None and data_code != None:
                        device = address_code
                        key_value = remote_controller_mapping.get(data_code)
                        print(f'id = {device} key_value = {key_value}')
                    else:
                        pass
                except Exception as e:
                    print(f'parse NEC error:{e}')

                finally:
                    pass


            # # 运行一定时间后退出循环，以避免无限循环  
            # if (time.time() - start_time) > 20:  # 例如，运行10秒钟  
            #     break  
    

                
    except:  
        pass  
    finally:  
        GPIO.cleanup()  




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
  
# 读取相机帧的线程函数  
def read_thread_worker():  
    global cap,frame_queue,key_value
    
    font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 10)
    text = "Hello, OpenCV! 你好"  
    position = (50, 100)  # 左上角坐标  
    font_color={255,0,0}
    
    # 记录开始时间和帧计数  
    start_time = time.time()  
    frame_count = 0 
    
 
    while True: 
        if key_value == "#":
            
            break

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
        if (time.time() - start_time) > 200:  # 例如，运行10秒钟  
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
    global cap,resize_queue,puttext_queue
    font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 30)
    text = "Hello, OpenCV! 你好"  
    position = (50, 100)  # 左上角坐标  
    font_color={255,0,0}
    
    start_time = time.time() 
    frame_count = 0  
    while True:
        if not resize_queue.empty():
            frame = resize_queue.get()
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
def display_thread_worker():  
    global cap,frame_queue,resize_queue,puttext_queue,key_value


    
    # cv2.namedWindow("Frame",cv2.WINDOW_FREERATIO) # 显示图片时：优先铺满，不保持原始图片长宽比
    cv2.namedWindow("Frame",cv2.WINDOW_KEEPRATIO)   # 显示图片时：保持原始图片长宽比，优先填充高，宽度不够两边留白
    cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print(cap.get(cv2.CAP_PROP_FPS))
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 获取FourCC  
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  
    # 将FourCC转换为对应的字符  
    fourcc_char = chr((fourcc >> 0) & 0xFF) + chr((fourcc >> 8) & 0xFF) + chr((fourcc >> 16) & 0xFF) + chr((fourcc >> 24) & 0xFF)  
    print(f"Current FourCC: {fourcc_char}") 



    
    
    # 记录开始时间和帧计数  
    start_time = time.time()  
    frame_count = 0  
    empty_count = 0

    while True:  
        if key_value == "#":
            break


        if not frame_queue.empty():  # 确保队列中有帧  
            frame = frame_queue.get() 
            cv2.imshow("Frame", frame) 

             
              
                        
            # # 等待按键，如果按下'q'键则退出循环  
            # if cv2.waitKey(1) & 0xFF == ord('q'):  
            #     break  
            
            key=cv2.waitKey(10)
            if key!=-1:
                print('key:',key)





            
            frame_count += 1 
            # print(f'frame_count out:{frame_count}') 
    
            
            
        else:
            time.sleep(0.001)
            empty_count+=1
            # print(f'empty:{empty_count}')
            pass
            
        # 运行一定时间后退出循环，以避免无限循环  
        if (time.time() - start_time) > 200:  # 例如，运行10秒钟  
            break  


    # 计算并打印估算的帧率  
    elapsed_time = time.time() - start_time  
    estimated_fps = frame_count / elapsed_time  
    print(f"out FPS: {estimated_fps:.2f} FRAMES:{frame_count}")
  


  
def init_camera():
    

    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise BaseException
    except BaseException:
        print(f'open camera failed')
    else:
        
        
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

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH,854)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)


        # cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc('M','J','P','G'))
        # cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter.fourcc('Y','U','Y','V'))
        # cap.set(cv2.CAP_PROP_FPS,30)






        return cap
    



        

  
if __name__ == "__main__":
    
    key_value=0


    frame_queue = Queue(maxsize=2)
    resize_queue = Queue(maxsize=2)
    puttext_queue = Queue(maxsize=2)

    cap = init_camera()


    # 添加边沿检测事件，当引脚从低电平变为高电平时调用回调函数  
    GPIO.add_event_detect(IR_PIN, GPIO.RISING, callback=rising)  
    
  
        


    # 创建并启动线程  
    read_thread = threading.Thread(target=read_thread_worker)  
    # resize_thread = threading.Thread(target=resize_thread_worker)
    # puttext_thread = threading.Thread(target=puttext_thread_worker)
    display_thread = threading.Thread(target=display_thread_worker)  
    # control_thread = threading.Thread(target=control_thread_worker)
    
    read_thread.start() 
    # resize_thread.start() 
    # puttext_thread.start()
    display_thread.start()  
    # control_thread.start()
    
    # 等待线程完成（这里只是简单地等待，实际上可能需要更复杂的同步机制）  
    read_thread.join() 
    # resize_thread.join()
    # puttext_thread.join()
    display_thread.join()
    # control_thread.join()

    
    
    # 清理GPIO
    GPIO.cleanup()
    # 释放资源并关闭窗口  
    cap.release()  
    cv2.destroyAllWindows()