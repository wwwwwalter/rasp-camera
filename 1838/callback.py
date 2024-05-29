import RPi.GPIO as GPIO  
import time  

  
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
        return
    
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
        
        if 1.8<data_time<5:
            keycode='1' + keycode
            print(f'[{rising_index:02d}][1]:{data_time}')
        elif 0<data_time<1.8:
            keycode='0' + keycode
            print(f'[{rising_index:02d}][0]:{data_time}')
        else:
            keycode='-' + keycode
            print(f'[{rising_index:02d}][-]:{data_time}')
            pass
        if rising_index == 34:
            rising_index = 0 
            print(f'keycode:{keycode}')
            # 校验数据并解析数据帧
            parse_keyvalue(keycode)
            
            

    
            
    

        
        
        
        

    


    
    
    

  
# 添加边沿检测事件，当引脚从高电平变为低电平时调用回调函数  
# GPIO.add_event_detect(IR_PIN, GPIO.FALLING, callback=falling)  
GPIO.add_event_detect(IR_PIN, GPIO.RISING, callback=rising)  
  
try:  
    # 无限循环，保持程序运行  
    while True:  
        time.sleep(1)  # 休眠，防止程序过快退出，实际上由边沿检测事件触发回调函数  
except KeyboardInterrupt:  
    # 如果用户按下Ctrl+C，则清理GPIO设置并退出  
    GPIO.cleanup()
    
# GPIO.wait_for_edge(channel, GPIO_RISING, timeout=5000)