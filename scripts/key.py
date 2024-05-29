import cv2  
  
# 打开相机，参数0表示默认相机  
cap = cv2.VideoCapture(0)  
  
# 检查相机是否成功打开  
if not cap.isOpened():  
    print("Error opening video stream or file")  
    exit()  
  
while True:  
    # 逐帧读取  
    ret, frame = cap.read()  
  
    if not ret:  
        print("Can't receive frame (stream end?). Exiting ...")  
        break  
  
    # 显示帧  
    cv2.imshow('Camera Feed', frame)  
  
    # 如果按下'q'键，则退出循环  
    # if cv2.waitKey(1) & 0xFF == ord('q'):  
    #     break  
    
    key = cv2.waitKey(1)
    
    if key != -1:
        print('key:',key)
  
# 释放相机资源  
cap.release()  
# 关闭所有OpenCV窗口  
cv2.destroyAllWindows()