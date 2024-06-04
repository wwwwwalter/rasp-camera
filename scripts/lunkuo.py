'''
Opencv中的轮廓：
demo1
'''
import cv2
 
img = cv2.imread('stack.jpg')
# 图像灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("1",gray)
# 3*3核的高斯滤波
gray = cv2.GaussianBlur(gray, (3, 3), 0)
# cv2.imshow("2",gray)
# canny边缘检测
gray = cv2.Canny(gray, 100, 300)
# cv2.imshow("3",gray)
 
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow("thresh1",thresh)
 
# binary是最后返回的二值图像
#findContours()第一个参数是源图像、第二个参数是轮廓检索模式，第三个参数是轮廓逼近方法
#输出是轮廓和层次结构，轮廓是图像中所有轮廓的python列表，每个单独的轮廓是对象边界点的(x,y)坐标的Numpy数组
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
print(hierarchy.dtype)  # 应该输出 int32  
print(hierarchy.shape)  # 应该输出类似 (n, 4)，其中 n 是轮廓的数量
cv2.imshow("thresh1", thresh)
cv2.drawContours(img, contours, -1, (0, 0, 255), 1,hierarchy=hierarchy,maxLevel=2)

cv2.imshow("img", img)

cv2.waitKey(10*1000)
 
