import cv2  
import numpy as np  
  
# 设定图像尺寸  
width, height = 1920, 1080  
  
# 创建一个全黑的RGBA图像（4通道，包括透明度）  
image = np.zeros((height, width, 4), dtype=np.uint8)  
  
# 设置圆的参数  
center_coordinates = (width // 2, height // 2)  
radius = height // 2  # 圆的半径为高度的一半，即540像素  
  
# 在图像中间绘制一个白色的圆（BGR全为255），但Alpha通道为0（完全透明）  
cv2.circle(image, center_coordinates, radius, (255, 255, 255, 0), thickness=-1)  
  
# 保存图像为PNG格式，保持透明度  
cv2.imwrite('transparent_circle.png', image)  
  
# 显示图像以验证结果  
cv2.imshow('Image with Transparent Circle', image)  
cv2.waitKey(0)  
cv2.destroyAllWindows()