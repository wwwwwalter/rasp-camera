    # if signal_off_time * 1000 > 4: # 这一帧是数据帧
    #     print(f'数据帧')
    #     # 地址码
    #     for index in range(8):  
    #         # 每个数据位有560us高电平脉冲（反向） 
    #         while GPIO.input(pin) == 0:  
    #             pass

    #         # 1  1.69ms低电平（反向）
    #         # 0  0.56ms低电平（反向）
    #         start_time = time.time()   
    #         while GPIO.input(pin) == 1:  
    #             pulse_off_time = time.time() - start_time

    #         if pulse_off_time * 1000 < 1:
    #             address_code = '0' + address_code
    #         else:
    #             address_code = '1' + address_code 

    #     # 地址码反码
    #     for index in range(8):  
    #         # 每个数据位有560us高电平脉冲（反向） 
    #         while GPIO.input(pin) == 0:  
    #             pass

    #         # 1  1.69ms低电平（反向）
    #         # 0  0.56ms低电平（反向）
    #         start_time = time.time()   
    #         while GPIO.input(pin) == 1:  
    #             pulse_off_time = time.time() - start_time

    #         if pulse_off_time * 1000 < 1:
    #             address_inversion = '0' + address_inversion
    #         else:
    #             address_inversion = '1' + address_inversion  
    

    #     # 数据码
    #     for index in range(8):
    #         # 每个数据位有560us高电平脉冲（反向）
    #         while GPIO.input(pin) == 0:  
    #             pass

    #         # 1  1.69ms低电平（反向）
    #         # 0  0.56ms低电平（反向）
    #         start_time = time.time()   
    #         while GPIO.input(pin) == 1:  
    #             pulse_off_time = time.time() - start_time
    #         if pulse_off_time * 1000 < 1:
    #             data_code = '0' + data_code
    #         else:
    #             data_code = '1' + data_code

    #     # 数据码反码
    #     for index in range(8):
    #         # 每个数据位有560us高电平脉冲（反向）
    #         while GPIO.input(pin) == 0:  
    #             pass

    #         # 1  1.69ms低电平（反向）
    #         # 0  0.56ms低电平（反向）
    #         start_time = time.time()   
    #         while GPIO.input(pin) == 1:  
    #             pulse_off_time = time.time() - start_time
    #         if pulse_off_time * 1000 < 1:
    #             data_inversion = '0' + data_inversion
    #         else:
    #             data_inversion = '1' + data_inversion


        





        # print(f'ac:',address_code)
        # print(f'ai:',address_inversion)
        # print(f'dc:',data_code)
        # print(f'di:',data_inversion)
        
        
        
# 设置GPIO模式为BCM  
GPIO.setmode(GPIO.BCM)  
  
# 设置红外接收器的GPIO引脚（这里以18为例，请根据实际情况修改）  
IR_PIN = 18  
GPIO.setup(IR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  

# 按键键值
key_value = ""



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

count_put = 0
def ir_receiver_nec_decode(pin):
    global count_put
    count_put+=1
    print(f'put:{count_put}')

    code=""                 # 帧数据
    address_code=""         # 地址码
    address_inversion=""    # 地址码反码
    data_code=""            # 数据码
    data_inversion=""       # 数据码反码




    signal_on_time = 0
    # 引导码高电平：9ms高电平(反向)
    start_time = time.time() 
    while GPIO.input(pin) == 0:
        signal_on_time = time.time() - start_time
        if signal_on_time * 1000 > 10:# 解析到非NEC协议信号直接退出函数,防止陷入死循环
            print('long')
            return None,None
    if signal_on_time * 1000 < 8:
        
        print('short')
        print((time.time() - start_time)*1000)
        return None,None
    

    
        
    # 引导码低电平：
        # 4.5ms低电平(反向)表示下面是数据帧
        # 2.5ms低电平(反向)表示下面是重复帧
    start_time = time.time()  
    while GPIO.input(pin) == 1:  
        signal_off_time = time.time() - start_time
        if signal_off_time * 1000 > 5:
            return None,None
    if signal_off_time * 1000 < 2:
        return None,None
    
    
    
        
    if signal_off_time * 1000 > 4: # 这一帧是数据帧
        print(f'数据帧')
        # 有效数据共32bit，采用左拼接
        for index in range(32):  
            # 每个数据位有0.56ms高电平脉冲（反向）
            start_time = time.time()
            while GPIO.input(pin) == 0:
                pulse_on_time = time.time() - start_time
                if pulse_on_time * 1000 > 1:
                    return None,None

            # 1  1.69ms低电平（反向）
            # 0  0.56ms低电平（反向）
            start_time = time.time()   
            while GPIO.input(pin) == 1:  
                pulse_off_time = time.time() - start_time
                if pulse_off_time * 1000 > 2:
                    return None,None

            if pulse_off_time * 1000 < 1:
                code = '0' + code
            else:
                code = '1' + code 
        
        print(f'data:{code}')
        
        
        # 停止位：0.56ms高电平（反向）
        start_time = time.time()
        while GPIO.input(pin) == 0:
            stop_on_time = time.time() - start_time
            if stop_on_time * 1000 > 1:
                return None,None
        
        # 解析（左拼接）
        # 例帧：10111010 01000101 11111111 00000000
        #       数据反码  数据码   地址反码  地址码
        data_inversion = code[0:8]
        data_code = code[8:16]
        address_inversion = code[16:24]
        address_code = code[24:32]
        
        print(address_code)
        print(address_inversion)
        print(data_code)
        print(data_inversion)
        
        
        # 校验地址码
        if not check_code(address_code,address_inversion):
            print('address code check false')
            return None,None
        # 校验数据码
        if not check_code(data_code,data_inversion):
            print('data code check false')
            return None,None
        # 校验通过
        return address_code,data_code

    else: # 这一帧是重复帧
        print(f'重复帧')
        # 停止位：0.56ms高电平(反向)
        start_time = time.time()
        while GPIO.input(pin) == 0:
            stop_on_time = time.time() - start_time
            if stop_on_time * 1000 > 1:
                return None,None
            
        # 97.94ms低电平（反向）
        start_time = time.time()
        while GPIO.input(pin) == 1:
            temp_off_time = time.time()-start_time
            if temp_off_time * 1000 > 100:
                return None,None
        if temp_off_time * 1000 < 95:
            return None,None
        
        
        # 重复帧不返回键值，如果需要长按功能，可以返回首帧键值
        return None,None