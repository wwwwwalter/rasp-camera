import RPi.GPIO as GPIO  
import time  
import binascii

  
# 设置GPIO模式为BCM  
GPIO.setmode(GPIO.BCM)  
  
# 设置红外接收器的GPIO引脚（这里以18为例，请根据实际情况修改）  
IR_PIN = 18  
GPIO.setup(IR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  





# 校验地址码或者数据码
def check_code(code,code_inversion):
    inverted_str = ''  
    for char in code:  
        if char == '0':  
            inverted_str += '1'  
        elif char == '1':  
            inverted_str += '0'  
        else:  
            raise ValueError("Invalid binary string. It should only contain '0' or '1'.")  
    if inverted_str == code_inversion:
        return True
    else:
        return False


# 红外NEC协议(协议定义和实际接收器针脚电平相反)
# 引导码|地址码|地址码反码|数据码|数据码反码|停止位
# 引导码：9ms(高)+4.5ms(低)
# 地址码，地址码反码，数据码，数据码反码：8bit*4[LSB小端]优先传输低字节
# 逻辑1：0.560ms(高)+1.690ms(低)=2.25ms
# 逻辑0：0.560ms(高)+0.560ms(低)=1.12ms
# 停止位：0.5625ms(低)
def nec_decode(pin):


    address_code=""         # 地址码
    address_inversion=""    # 地址码反码
    data_code=""            # 数据码
    data_inversion=""       # 数据码反码

    print(f'---------------{"head"}----------------')
    # 检查起始码后的高电平（反向）
    start_time = time.time()  
    while GPIO.input(pin) == 0:  
        signal_on_time = time.time() - start_time  
    print('signal_on_time:',signal_on_time * 1000)

    # 检查起始码后的低电平（反向）
    start_time = time.time()  
    while GPIO.input(pin) == 1:  
        signal_off_time = time.time() - start_time  
    print('signal_off_time:',signal_off_time * 1000)

 
    # 地址码
    for index in range(8):  
        print(f'---------------{index}----------------')
        # 每个数据位有560us高电平脉冲（反向）
        start_time = time.time()   
        while GPIO.input(pin) == 0:  
            pulse_on_time = time.time() - start_time  
        print('pulse_on_time:',pulse_on_time * 1000)

        # 1:1690us低电平（反向）
        # 0:560us低电平（反向）
        start_time = time.time()   
        while GPIO.input(pin) == 1:  
            pulse_off_time = time.time() - start_time
        print('pulse_off_time:',pulse_off_time * 1000)

        if pulse_off_time * 1000 < 1:
            address_code = '0' + address_code
        else:
            address_code = '1' + address_code 

    # 地址码反码
    for index in range(8):  
        print(f'---------------{index}----------------')
        # 每个数据位有560us高电平脉冲（反向）
        start_time = time.time()   
        while GPIO.input(pin) == 0:  
            pulse_on_time = time.time() - start_time  
        print('pulse_on_time:',pulse_on_time * 1000)

        # 1:1690us低电平（反向）
        # 0:560us低电平（反向）
        start_time = time.time()   
        while GPIO.input(pin) == 1:  
            pulse_off_time = time.time() - start_time
        print('pulse_off_time:',pulse_off_time * 1000)

        if pulse_off_time * 1000 < 1:
            address_inversion = '0' + address_inversion
        else:
            address_inversion = '1' + address_inversion  
 

    # 数据码
    for index in range(8):
        print(f'---------------{index}----------------')
        # 每个数据位有560us高电平脉冲（反向）
        start_time = time.time()   
        while GPIO.input(pin) == 0:  
            pulse_on_time = time.time() - start_time  
        print('pulse_on_time:',pulse_on_time * 1000)  

        # 1:1690us低电平（反向）
        # 0:560us低电平（反向）
        start_time = time.time()   
        while GPIO.input(pin) == 1:  
            pulse_off_time = time.time() - start_time
        print('pulse_off_time:',pulse_off_time * 1000) 
        if pulse_off_time * 1000 < 1:
            data_code = '0' + data_code
        else:
            data_code = '1' + data_code

    # 数据码反码
    for index in range(8):
        print(f'---------------{index}----------------')
        # 每个数据位有560us高电平脉冲（反向）
        start_time = time.time()   
        while GPIO.input(pin) == 0:  
            pulse_on_time = time.time() - start_time  
        print('pulse_on_time:',pulse_on_time * 1000)  

        # 1:1690us低电平（反向）
        # 0:560us低电平（反向）
        start_time = time.time()   
        while GPIO.input(pin) == 1:  
            pulse_off_time = time.time() - start_time
        print('pulse_off_time:',pulse_off_time * 1000) 
        if pulse_off_time * 1000 < 1:
            data_inversion = '0' + data_inversion
        else:
            data_inversion = '1' + data_inversion


     
    # 停止位：0.5625ms高电平（反向）
    print(f'---------------{"stop"}----------------')
    start_time = time.time()   
    while GPIO.input(pin) == 0:  
        stop_on_time = time.time() - start_time  
    print('stop_on_time:',stop_on_time * 1000) 

    print(f'ac:',address_code)
    print(f'ai:',address_inversion)
    print(f'dc:',data_code)
    print(f'di:',data_inversion)

    

    # # 将二进制字符串转换为字节字符串  
    # bytes_string = bytearray()
    # for i in range(0, len(data), 8):  
    #     byte = data[i:i+8]  
    #     bytes_string.append(int(byte, 2))
    # # 将字节字符串转换为十六进制字符串  
    # hex_string = binascii.hexlify(bytes_string).decode('utf-8')  
    # print(hex_string)

    if not check_code(address_code,address_inversion):
        print('address code check false')
        return None,None
    
    if not check_code(data_code,data_inversion):
        print('data code check false')
        return None,None

    
    return address_code,data_code

    


  
    
      
    # # 等待起始码  
    # while GPIO.input(pin) == 0:  
    #     count = count + 1
    #     print(count,"low:",pin)
    #     signal_off = time.time()  
          
    # while GPIO.input(pin) == 1:  
    #     count = count + 1
    #     print(count,"high:",pin)
    #     signal_on = time.time()  
          
    # time_passed = (signal_on - signal_off) * 1e6  # 转换为微秒  
      
    # # 检查起始码  
    # if time_passed > HEADER_MARK - 200 and time_passed < HEADER_MARK + 200:  
    #     time.sleep(HEADER_SPACE / 1e6)  # 等待起始码后的间隔  
          
    #     data = 0  
    #     for _ in range(32):  # NEC协议通常包含32位数据  
    #         time.sleep(BIT_MARK / 1e6)  # 等待数据位的高电平  
              
    #         # 测量低电平时间来判断是'0'还是'1'  
    #         signal_off = time.time()  
    #         while GPIO.input(pin) == 0:  
    #             signal_on = time.time()  
                  
    #         time_passed = (signal_on - signal_off) * 1e6  
              
    #         if time_passed > ONE_SPACE - 200 and time_passed < ONE_SPACE + 200:  
    #             data = (data << 1) | 1  # 接收到的是'1'  
    #         elif time_passed > ZERO_SPACE - 200 and time_passed < ZERO_SPACE + 200:  
    #             data <<= 1  # 接收到的是'0'  
    #         else:  
    #             return None  # 数据位时间不符合协议规范，返回None  
              
    #     return data  
    # else:  
    #     return None  # 起始码不符合协议规范，返回None  
  




# 测试函数  
def test_ir_receiver():  
    try:  
        while True:  
            if GPIO.input(IR_PIN)==0:
                print('接受到遥控器信号')
                address_code,data_code = nec_decode(IR_PIN)
                print(address_code)
                print(data_code)
                
                break

    except KeyboardInterrupt:  
        pass  
    finally:  
        GPIO.cleanup()  
  
# 运行测试函数  
test_ir_receiver()