import RPi.GPIO as GPIO  
import time  
  
# 设置GPIO引脚编号模式为BCM  
GPIO.setmode(GPIO.BCM)  
  
# 设置红外接收模块的GPIO引脚为输入模式，并启用上拉电阻  
IR_PIN = 18  
GPIO.setup(IR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  
  
def decode_ir_signal():  
    # 这里可以添加解码红外信号的代码  
    # 通常涉及检测GPIO引脚上的电平变化，并解析出具体的红外码值  
    # NEC协议的起始码通常是9ms的高电平和4.5ms的低电平  
    start_time = time.time()  
    while GPIO.input(IR_PIN) == 0:  
        signal_on_time = time.time() - start_time  
        if signal_on_time > 0.009:  # 9ms以上的高电平可能表示起始码的开始  
            break  
    print('9ms')
      
    # 检查起始码后的低电平  
    start_time = time.time()  
    while GPIO.input(IR_PIN) == 1:  
        signal_off_time = time.time() - start_time  
        if signal_off_time > 0.004:  # 4.5ms以上的低电平  
            break  
    print('4ms')
      
    # 读取数据位和停止位  
    data = 0  
    for _ in range(32):  # NEC协议通常有32位数据  
        start_time = time.time()  
        # 等待下一个脉冲的开始  
        while GPIO.input(IR_PIN) == 0:  
            pulse_start_time = time.time()  
            if pulse_start_time - start_time > 0.01:  # 超时检查  
                return None  # 没有接收到有效的脉冲  
          
        # 测量脉冲宽度  
        while GPIO.input(IR_PIN) == 1:  
            pulse_width = time.time() - pulse_start_time  
            if pulse_start_time - start_time > 0.01:  # 超时检查  
                return None  # 没有接收到有效的脉冲  
          
        # 根据脉冲宽度判断是0还是1（NEC协议中，短脉冲代表0，长脉冲代表1）  
        if pulse_width < 0.006:  # 短脉冲  
            data <<= 1  # 左移一位，相当于乘以2  
        else:  # 长脉冲  
            data = (data << 1) | 1  # 左移一位后最低位设置为1  
      
    # 验证停止位（通常为长时间的高电平）  
    start_time = time.time()  
    while GPIO.input(IR_PIN) == 1:  
        if time.time() - start_time > 0.01:  # 停止位通常是一个较长的高电平  
            break  
      
    return data  # 返回解码后的数据   
  
try:  
    print("等待红外信号...")  
    while True:  
        if GPIO.input(IR_PIN) == 0:  
            print("接收到红外信号")  
            signal = decode_ir_signal()  
            print(signal)
            time.sleep(0.2)  # 防止重复检测同一信号  
except KeyboardInterrupt:  
    print("程序被中断")  
finally:  
    GPIO.cleanup()  # 清理GPIO设置，以便下次使用