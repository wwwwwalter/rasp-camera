# from pynput.keyboard import Key, Listener  
  
# def on_press(key):  
#     try:  
#         print('alphanumeric key {0} pressed'.format(  
#             key.char))  
#     except AttributeError:  
#         print('special key {0} pressed'.format(  
#             key))  
  
# def on_release(key):  
#     print('{0} release'.format(  
#         key))  
#     if key == Key.esc:  
#         # 停止监听  
#         return False  
  
# # 创建一个监听器  
# with Listener(on_press=on_press, on_release=on_release) as listener:  
#     listener.join()



# import evdev  

# # 替换为你的遥控器设备文件路径  
# device_path = '/dev/input/by-id/usb-0627_2.4G_Composite_Devic-event-kbd'  

# # 打开设备  
# dev = evdev.InputDevice(device_path)  

# print(dev)  # 打印设备信息  

# # 循环读取事件  
# for event in dev.read_loop():  
#     if event.type == evdev.ecodes.EV_KEY:  
#         print(f"Key {evdev.ecodes.KEY[event.code]:<30} {'pressed' if event.value else 'released'}")  

# # 注意：上面的read_loop是一个无限循环，你可能需要一种方式来优雅地退出它（例如，通过捕获SIGINT）