import cv2  
  
# 创建一个VideoCapture对象，参数为0表示使用默认的相机  
cap = cv2.VideoCapture(0)  
  
# 检查相机是否成功打开  
if not cap.isOpened():  
    print("Error opening video stream or file")  
    exit()  
  
while True:  
    # 逐帧捕获  
    ret, frame = cap.read()  
  
    if not ret:  
        print("Can't receive frame (stream end?). Exiting ...")  
        break  
  
    # 显示结果帧  
    cv2.imshow('frame', frame)  
  
    # # 如果按下'q'键，则退出循环  
    # if cv2.waitKey(1) & 0xFF == ord('q'):  
    #     break  
    
    key=cv2.waitKey(1)
    if key!=-1:
        print(key)
  
# 完成后，释放捕获并销毁所有窗口  
cap.release()  
cv2.destroyAllWindows()