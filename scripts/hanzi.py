from PIL import Image, ImageFont, ImageDraw  
import cv2  
import numpy as np  

# 读取OpenCV图像并转换为PIL格式  
img = cv2.imread('image.jpg')  
cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
pilimg = Image.fromarray(cv2img)  

# 创建PIL的绘图对象和字体对象  
draw = ImageDraw.Draw(pilimg)  
font = ImageFont.truetype("simsun.ttf", 40, encoding="utf-8")  

# 在图像上添加中文字符  
draw.text((50, 50), "你好", font=font, fill=(255, 0, 0))  

# 转换回OpenCV格式并显示  
cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)  
cv2.imshow("image", cv2charimg)  
cv2.waitKey(0)