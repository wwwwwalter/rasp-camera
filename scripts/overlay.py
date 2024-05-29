import cv2  
import numpy as np  
  
# 打开相机  
cap = cv2.VideoCapture(0)  
  
while True:  
    # 读取相机帧  
    ret, frame = cap.read()  
    if not ret:  
        print("无法接收帧 (流可能已结束？)")  
        break  
  
    # 获取帧的高度和宽度  
    height, width = frame.shape[:2]  
  
    # 创建一个与帧大小相同的透明图层（BGR）  
    overlay = np.zeros((height, width, 3), dtype=np.uint8)  
  
    # 在透明图层上添加文字信息  
    # 注意：OpenCV不支持真正的透明文本，所以我们只是降低文字颜色的亮度来模拟透明效果  
    font = cv2.FONT_HERSHEY_SIMPLEX  
    fontScale = 1  
    fontColor = (0, 255, 0)  # BGR颜色，这里使用绿色  
    thickness = 2  
    text = "Hello, OpenCV!"  
    textSize, baseline = cv2.getTextSize(text, font, fontScale, thickness)  
    textX = 10  
    textY = height - 10 - baseline  
    cv2.putText(overlay, text, (textX, textY), font, fontScale, fontColor, thickness)  
  
    # 叠加透明图层到相机帧上（这里只是简单地叠加，没有真正的透明度）  
    # 如果你想要透明度效果，你可能需要更复杂的处理或使用其他库（如PIL）  
    frame_with_overlay = cv2.addWeighted(frame, 1, overlay, 0.8, 0)  # 这里0.5是模拟的透明度（叠加权重）  
  
    # 显示叠加后的图像  
    cv2.imshow('Camera with overlay', frame_with_overlay)  
  
    # 按'q'键退出循环  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
  
# 释放相机并关闭所有OpenCV窗口  
cap.release()  
cv2.destroyAllWindows()