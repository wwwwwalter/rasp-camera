import cv2  
  
# 打开相机，0表示使用默认相机  
cap = cv2.VideoCapture(0)  
  
# 检查相机是否成功打开  
if not cap.isOpened():  
    print("无法打开相机")  
    exit()  
  
# 创建一个窗口来显示视频  
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)  

    # cv2.namedWindow("Frame",cv2.WINDOW_FREERATIO) # 显示图片时：优先铺满，不保持原始图片长宽比
cv2.namedWindow("Video",cv2.WINDOW_KEEPRATIO)   # 显示图片时：保持原始图片长宽比，优先填充高，宽度不够两边留白
cv2.setWindowProperty("Video",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
  
while True:  
    # 读取一帧图像  
    ret, frame = cap.read()  
  
    # 如果读取成功，ret为True  
    if not ret:  
        print("无法读取视频流")  
        break  
  
    # 在窗口中显示图像  
    cv2.imshow("Video", frame)  
  
    # 等待按键，如果按下'q'键则退出循环  
    # if cv2.waitKey(1) & 0xFF == ord('q'):  
    #     break  
    
    key=cv2.waitKey(1)
    if(key != -1):
        print("key:",key)
  
# 释放相机资源  
cap.release()  
# 销毁所有窗口  
cv2.destroyAllWindows()